#!/usr/bin/env python
"""P0（严格独立 Holdout）· 抓取**从未参与开发**的公开 ROCOv2 原始数据，建立新的独立测试集。

背景：此前封存的 55 条 holdout 经审计已被旧版 P3 困难样本挖掘/难负例构造使用（14/55 与
hard_cases 重叠，55/55 出现在 hard_negatives 候选池），因此**不是真正未见数据**，不得再称为 holdout。

本脚本：
1. 从 HF datasets-server /rows 抓取 test split 中**未被本项目摄取过**的 row（按 image_id + 图像
   SHA256 双重去重，排除与既有 1000 条重复者）。
2. 图像存到独立目录 data/raw/images_holdoutv2/（不混入原 data/raw/images，避免污染开发集）。
3. 输出 data/metadata/holdout_v2_manifest.jsonl，并记录近重复簇分组所需字段（pHash 后续计算）。

注意：ROCOv2 公开接口不暴露来源论文 ID，因此"按来源论文分组"不可行；改用**近重复簇分组**
（pHash + 图注词面相似），确保独立测试集内部及与开发集之间无近重复泄漏。此限制如实记录。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_IMG = ROOT / "data/raw/images_holdoutv2"
OUT_META = ROOT / "data/metadata"
OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_META.mkdir(parents=True, exist_ok=True)

DATASET = "eltorio/ROCOv2-radiology"
BASE = "https://datasets-server.huggingface.co/rows"
UA = {"User-Agent": "MedAlignFlywheel-audit/1.0"}


def jl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def get_json(url: str, retries: int = 4):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            wait = 5 * (i + 1)
            print(f"    retry {i+1}/{retries} after {wait}s ({e})", flush=True)
            time.sleep(wait)
    return None


def download(url: str, retries: int = 3) -> bytes | None:
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            time.sleep(3 * (i + 1))
    return None


def main() -> int:
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    start_offset = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

    # 既有样本：按 sample_id 与图像 sha256 去重
    existing = jl(ROOT / "data/metadata/raw_manifest.jsonl")
    seen_ids = {r["sample_id"] for r in existing}
    seen_sha = {r.get("image_sha256") for r in existing if r.get("image_sha256")}
    print(f"[init] existing ids={len(seen_ids)} sha={len(seen_sha)}", flush=True)

    out_rows: list[dict] = []
    offset = start_offset
    max_offset = 9900
    fetched = 0

    while len(out_rows) < target and offset < max_offset:
        length = 100
        url = (f"{BASE}?dataset={urllib.parse.quote(DATASET)}&config=default"
               f"&split=test&offset={offset}&length={length}")
        data = get_json(url)
        if not data or "rows" not in data:
            print(f"[warn] no rows at offset {offset}", flush=True)
            offset += length
            time.sleep(2)
            continue
        rows = data["rows"]
        if not rows:
            break
        for row in rows:
            if len(out_rows) >= target:
                break
            r = row["row"]
            img = r.get("image") or {}
            src = img.get("src")
            image_id = r.get("image_id")
            caption = r.get("caption") or ""
            cui = r.get("cui") or []
            if not src or not image_id or not caption.strip():
                continue
            sid = str(image_id)
            if sid in seen_ids:
                continue  # 已参与开发，跳过
            blob = download(src)
            if not blob:
                continue
            sha = hashlib.sha256(blob).hexdigest()
            if sha in seen_sha:
                continue  # 图像重复（可能来自同论文），跳过
            seen_sha.add(sha)
            fp = OUT_IMG / f"{sid}.jpg"
            fp.write_bytes(blob)
            out_rows.append({
                "sample_id": sid,
                "split": "holdout_v2",
                "caption": caption,
                "cui": list(cui),
                "image_relpath": f"images_holdoutv2/{sid}.jpg",
                "image_sha256": sha,
                "image_bytes": len(blob),
                "row_idx": row.get("row_idx"),
                "source_split_url": f"hf-rows:{DATASET}:test:{row.get('row_idx')}",
            })
            fetched += 1
            if fetched % 50 == 0:
                print(f"    fetched={fetched} kept={len(out_rows)} offset={offset}", flush=True)
        offset += length
        time.sleep(1.2)  # 礼貌限速

    path = OUT_META / "holdout_v2_manifest.jsonl"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"[done] kept {len(out_rows)} new independent samples -> {path}", flush=True)
    print(f"[done] images in {OUT_IMG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
