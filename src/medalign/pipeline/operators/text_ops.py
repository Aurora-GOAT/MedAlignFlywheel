"""图注文本质量算子。

规则全部是字面的、**可被第三方重算**的，不代表医学正确性。
特别注意：本轮数据集是论文图注，短图注不等于"没价值"，
因此对短图注采用 warn+reject 两级阈值，并在报告里披露被拒样本。
"""

from __future__ import annotations

from ..operators.base import OK, WARN, REJECT, Finding, OpResult, Operator
from ...common.textnorm import (
    caption_fingerprint,
    digit_ratio,
    has_nonprintable,
    has_placeholder,
    nonascii_ratio,
    strip_panel_markers,
    word_tokens,
)

OP_VERSION = "text-quality-v1.1"

DEFAULT_THRESHOLDS = {
    "min_tokens_reject": 3,
    "min_tokens_warn": 6,
    "max_nonascii": 0.20,
    "max_digit_ratio": 0.35,
    "max_repeat_same_token_ratio": 0.45,
}


class CaptionQualityOperator(Operator):
    name = "caption_quality"
    version = OP_VERSION
    proxy_only = True

    def __init__(self, thresholds: dict | None = None, duplicate_index: dict | None = None):
        self.t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        # {caption_fingerprint -> [sample_id, ...]}，由 runner 预计算
        self.dup_index = duplicate_index or {}

    def run(self, record: dict, ctx: dict) -> OpResult:
        sid = record["sample_id"]
        res = OpResult(sample_id=sid)
        caption = record.get("caption")
        if caption is None:
            res.findings.append(
                Finding(self.name, self.version, REJECT, "CAPTION_MISSING", "图注字段缺失")
            )
            return res

        caption = str(caption)
        res.metrics["caption_raw_len"] = len(caption)
        res.metrics["caption_n_tokens"] = len(word_tokens(caption))

        if not caption.strip():
            res.findings.append(
                Finding(self.name, self.version, REJECT, "CAPTION_EMPTY", "图注为空字符串")
            )
            return res

        if has_nonprintable(caption):
            res.findings.append(
                Finding(self.name, self.version, REJECT, "CAPTION_NONPRINTABLE",
                        "图注含不可打印控制字符", evidence={"snippet": repr(caption[:120])})
            )

        if has_placeholder(caption):
            res.findings.append(
                Finding(self.name, self.version, REJECT, "CAPTION_PLACEHOLDER",
                        "图注含占位/模板噪声", evidence={"snippet": caption[:120]})
            )

        # 去掉 "(a)"、"Panel B" 之类结构标记后是否还有内容
        stripped = strip_panel_markers(caption)
        res.metrics["caption_stripped_len"] = len(stripped)
        if len(stripped.strip()) == 0:
            res.findings.append(
                Finding(self.name, self.version, REJECT, "CAPTION_ONLY_PANEL_MARKER",
                        "图注去掉版面标记后为空", evidence={"raw": caption[:120]})
            )
            return res

        n_tok = len(word_tokens(stripped))
        res.metrics["caption_effective_tokens"] = n_tok
        if n_tok < self.t["min_tokens_reject"]:
            res.findings.append(
                Finding(self.name, self.version, REJECT, "CAPTION_TOO_SHORT",
                        "有效词数过少，几乎无检索可判别信息",
                        evidence={"n_tokens": n_tok, "text": stripped[:120]},
                        threshold=self.t["min_tokens_reject"])
            )
        elif n_tok < self.t["min_tokens_warn"]:
            res.findings.append(
                Finding(self.name, self.version, WARN, "CAPTION_SHORT",
                        "有效词数偏少", evidence={"n_tokens": n_tok},
                        threshold=self.t["min_tokens_warn"])
            )

        na = nonascii_ratio(caption)
        res.metrics["caption_nonascii_ratio"] = round(na, 4)
        if na > self.t["max_nonascii"]:
            res.findings.append(
                Finding(self.name, self.version, WARN, "CAPTION_HIGH_NONASCII",
                        "非 ASCII 字符占比偏高", evidence={"ratio": round(na, 3)},
                        threshold=self.t["max_nonascii"])
            )

        dr = digit_ratio(caption)
        res.metrics["caption_digit_ratio"] = round(dr, 4)
        if dr > self.t["max_digit_ratio"]:
            res.findings.append(
                Finding(self.name, self.version, WARN, "CAPTION_HIGH_DIGIT",
                        "数字占比偏高（可能是编号/测量值列表）",
                        evidence={"ratio": round(dr, 3)}, threshold=self.t["max_digit_ratio"])
            )

        # 单词重复度：检测 "CT CT CT" 这类抽取异常
        toks = word_tokens(stripped)
        if toks:
            from collections import Counter
            top_n = Counter(toks).most_common(1)[0][1]
            rep = top_n / len(toks)
            res.metrics["caption_top_token_ratio"] = round(rep, 4)
            if rep > self.t["max_repeat_same_token_ratio"]:
                res.findings.append(
                    Finding(self.name, self.version, WARN, "CAPTION_REPETITIVE",
                            "同一词重复占比过高", evidence={"ratio": round(rep, 3)},
                            threshold=self.t["max_repeat_same_token_ratio"])
                )

        # 跨样本重复图注（同样的图注被配到不同图 → 检索评测中会产生"假负例"）
        fp = caption_fingerprint(stripped)
        res.metrics["caption_fingerprint"] = fp
        owners = self.dup_index.get(fp, [])
        if len(owners) > 1:
            res.findings.append(
                Finding(self.name, self.version, WARN, "CAPTION_DUPLICATE_ACROSS_SAMPLES",
                        "同一图注指纹出现在多个样本上，可能在检索评测中形成假负例",
                        evidence={"peer_sample_ids": [o for o in owners if o != sid][:5],
                                  "n_owners": len(owners)})
            )

        if not res.rejected and not res.warnings:
            res.findings.append(
                Finding(self.name, self.version, OK, "CAPTION_OK", "通过全部图注质量检查")
            )
        return res
