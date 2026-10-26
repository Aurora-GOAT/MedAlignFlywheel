"""图像完整性算子。

检查项（全部是**可解释的字面规则**）：
- IMAGE_DECODE_FAIL    解码失败（不是合法图像）
- IMAGE_TOO_SMALL      短边小于阈值（默认 64px）
- IMAGE_TINY_FILE      文件字节过小（默认 <2KB，通常是占位图或缩略图）
- IMAGE_NEAR_BLANK     灰度标准差极低或灰度级数极少 → 疑似空白/单色图
- IMAGE_ASPECT_EXTREME 长宽比超过阈值（可能是拼接图或排版错误）
- IMAGE_MODE_UNUSUAL   非常见色彩模式（会在下游统一转成 RGB，仅记录）
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from ..operators.base import OK, REJECT, WARN, Finding, OpResult, Operator
from ...common.perceptual import dhash, image_stats, phash, _resample_lanczos

OP_VERSION = "image-integrity-v1.1"

DEFAULT_THRESHOLDS = {
    "min_short_side": 64,
    "min_short_side_warn": 128,
    "min_bytes": 2048,
    "min_gray_std": 3.0,
    "min_gray_std_warn": 8.0,
    "min_unique_levels": 8,
    "max_aspect_ratio": 8.0,
    "max_aspect_ratio_hard": 15.0,
}


class ImageIntegrityOperator(Operator):
    name = "image_integrity"
    version = OP_VERSION
    proxy_only = True

    def __init__(self, thresholds: dict | None = None):
        self.t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    def run(self, record: dict, ctx: dict) -> OpResult:
        sid = record["sample_id"]
        res = OpResult(sample_id=sid)
        raw_root = Path(ctx["raw_root"])
        rel = record.get("image_relpath")
        if not record.get("image_available") or not rel:
            res.findings.append(
                Finding(
                    self.name, self.version, REJECT, "IMAGE_MISSING",
                    "缺少图像文件或下载失败",
                    evidence={"image_available": record.get("image_available"),
                              "download_error": record.get("download_error")},
                )
            )
            return res

        path = raw_root / rel
        if not path.exists():
            res.findings.append(
                Finding(self.name, self.version, REJECT, "IMAGE_MISSING",
                        "清单声明存在但磁盘上找不到文件", evidence={"path": str(path)})
            )
            return res

        data = path.read_bytes()
        n_bytes = len(data)
        res.metrics["image_bytes"] = n_bytes

        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except Exception as exc:  # noqa: BLE001
            res.findings.append(
                Finding(self.name, self.version, REJECT, "IMAGE_DECODE_FAIL",
                        f"图像解码失败: {type(exc).__name__}",
                        evidence={"error": str(exc)[:200]}, threshold=None)
            )
            return res

        w, h = img.size
        gray = image_stats(img)
        res.metrics.update(
            {
                "width": w,
                "height": h,
                "aspect_ratio": round(max(w, h) / max(1, min(w, h)), 3),
                "gray_std": round(gray["std"], 3),
                "gray_mean": round(gray["mean"], 3),
                "gray_unique_levels": gray["unique_levels"],
                "mode": img.mode,
            }
        )

        if n_bytes < self.t["min_bytes"]:
            res.findings.append(
                Finding(self.name, self.version, WARN, "IMAGE_TINY_FILE",
                        "图像文件字节数过小，可能是占位图/缩略图",
                        evidence={"bytes": n_bytes}, threshold=self.t["min_bytes"])
            )

        short_side = min(w, h)
        if short_side < self.t["min_short_side"]:
            res.findings.append(
                Finding(self.name, self.version, REJECT, "IMAGE_TOO_SMALL",
                        "短边低于阈值，难以作为视觉输入",
                        evidence={"short_side": short_side, "size": [w, h]},
                        threshold=self.t["min_short_side"])
            )
        elif short_side < self.t["min_short_side_warn"]:
            res.findings.append(
                Finding(self.name, self.version, WARN, "IMAGE_SMALL_SIDE",
                        "短边偏小", evidence={"short_side": short_side},
                        threshold=self.t["min_short_side_warn"])
            )

        ar = max(w, h) / max(1, min(w, h))
        if ar > self.t["max_aspect_ratio_hard"]:
            res.findings.append(
                Finding(self.name, self.version, REJECT, "IMAGE_ASPECT_EXTREME",
                        "长宽比极端，可能是拼接条带或排版错误",
                        evidence={"aspect_ratio": round(ar, 2)},
                        threshold=self.t["max_aspect_ratio_hard"])
            )
        elif ar > self.t["max_aspect_ratio"]:
            res.findings.append(
                Finding(self.name, self.version, WARN, "IMAGE_ASPECT_SKEW",
                        "长宽比偏大", evidence={"aspect_ratio": round(ar, 2)},
                        threshold=self.t["max_aspect_ratio"])
            )

        if gray["std"] < self.t["min_gray_std"] or gray["unique_levels"] < self.t["min_unique_levels"]:
            res.findings.append(
                Finding(self.name, self.version, REJECT, "IMAGE_NEAR_BLANK",
                        "灰度标准差或灰度级数过低，疑似空白/纯色图",
                        evidence={"gray_std": round(gray["std"], 3),
                                  "unique_levels": gray["unique_levels"]},
                        threshold={"std": self.t["min_gray_std"],
                                   "levels": self.t["min_unique_levels"]})
            )
        elif gray["std"] < self.t["min_gray_std_warn"]:
            res.findings.append(
                Finding(self.name, self.version, WARN, "IMAGE_LOW_CONTRAST",
                        "对比度偏低", evidence={"gray_std": round(gray["std"], 3)},
                        threshold=self.t["min_gray_std_warn"])
            )

        if img.mode not in ("RGB", "L", "RGBA", "P", "1", "LA", "CMYK", "I;16"):
            res.findings.append(
                Finding(self.name, self.version, WARN, "IMAGE_MODE_UNUSUAL",
                        "非常见色彩模式", evidence={"mode": img.mode})
            )

        # 感知哈希：即使在 reject 的情况下也计算，供后续近重复 / 泄漏检查使用
        try:
            res.metrics["dhash64"] = dhash(img)
            res.metrics["phash64"] = phash(img)
        except Exception as exc:  # noqa: BLE001
            res.findings.append(
                Finding(self.name, self.version, WARN, "IMAGE_HASH_FAIL",
                        f"感知哈希计算失败: {type(exc).__name__}", evidence={"error": str(exc)[:120]})
            )

        if not res.rejected and not res.warnings:
            res.findings.append(
                Finding(self.name, self.version, OK, "IMAGE_OK", "通过全部图像完整性检查")
            )
        return res
