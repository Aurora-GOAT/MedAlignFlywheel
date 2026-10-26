#!/usr/bin/env python
"""P2：跨模态检索评测（真实 BiomedCLIP 向量 + 冻结 test 子集）。

产出 reports/retrieval_metrics.json：
- I2T / T2I 的 Recall@1/5/10、MRR、中位排名 + Bootstrap 95% 区间；
- 正例 vs 错配的相似度分布，以及**净化后**的 AUROC；
- 按影像模态 / 图注长度 / 概念长尾的分层错误分析；
- 逐样本排名（供 P3 难例挖掘复用）。

关于"负例是不是真的错"
-----------------------
ROCOv2 里同一篇论文的多张子图常常共享几乎相同的图注，此时把另一个样本
当成"错误答案"并不公平。这里对每个错配 pair 做可核验的模糊判定：
图注词包含率过高、或图像 pHash 距离过小 → 标记为 ambiguous，
**不计入**负例统计，但会如实公开被排除的数量。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from medalign.common.stats import paired_bootstrap_diff  # noqa: E402
from medalign.common.textnorm import token_set  # noqa: E402
from medalign.eval.retrieval import (  # noqa: E402
    RetrievalConfig,
    _ranks_from_scores,
    auditritt,
    bootstrap_recall,
    classify_negatives,
    similarity_matrix,
    stratified_metrics,
)


def load_manifest(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def eval_split(split: str, cache_dir: str, manifest: list[dict], cfg: RetrievalConfig) -> tuple[dict, list[dict]]:
    cache = Path(cache_dir)
    data = np.load(cache / f"emb_{split}.npz")
    ids = json.loads((cache / f"emb_{split}.ids.json").read_text(encoding="utf-8"))
    by_id = {r["sample_id"]: r for r in manifest if r.get("split") == split}

    order = [i for i, sid in enumerate(ids) if sid in by_id]
    ids = [ids[i] for i in order]
    img_emb = data["image_emb"][order]
    txt_emb = data["text_emb"][order]
    n = len(ids)
    cfg.candidate_size = n

    sim = similarity_matrix(img_emb, txt_emb)
    i2t_ranks = _ranks_from_scores(sim)
    t2i_ranks = _ranks_from_scores(sim.T)
    ks = list(cfg.ks)

    metrics: dict = {"n_candidates": n, "i2t": {}, "t2i": {}}
    for name, ranks in (("i2t", i2t_ranks), ("t2i", t2i_ranks)):
        entry = {f"recall@{k}": float((ranks <= k).mean()) for k in ks}
        entry["mrr"] = float((1.0 / ranks).mean())
        entry["median_rank"] = float(np.median(ranks))
        entry["bootstrap"] = bootstrap_recall(ranks, ks, n_boot=cfg.n_boot, seed=cfg.seed)
        metrics[name] = entry

    paired = {}
    for k in ks:
        pt, lo, hi = paired_bootstrap_diff(
            (i2t_ranks <= k).astype(float),
            (t2i_ranks <= k).astype(float),
            n_boot=cfg.n_boot,
            seed=cfg.seed,
        )
        paired[f"recall@{k}"] = {"diff": pt, "lo": lo, "hi": hi}
    metrics["i2t_minus_t2i_paired"] = paired

    # 负例净化 + 分布 + AUROC
    caption_tokens = [token_set(by_id[s]["caption"]) for s in ids]
    hashes = None
    try:
        hashes = [int(by_id[s]["metrics"]["phash64"]) for s in ids]
    except (KeyError, TypeError):
        hashes = None
    masks = classify_negatives(sim, caption_tokens, hashes, cfg)
    metrics["negative_audit"] = auditritt(sim, masks, cfg)

    labels = {
        "modality_proxy": [by_id[s]["modality_proxy"]["modality"] for s in ids],
        "caption_length_band": [by_id[s]["complexity"]["complexity_band"] for s in ids],
        "concept_group": [
            "longtail_all_singleton"
            if (by_id[s]["metrics"].get("cui_singleton_list")
                and len(by_id[s]["metrics"]["cui_singleton_list"])
                == by_id[s]["metrics"].get("n_cui_unique", 0))
            else "has_shared_concept"
            for s in ids
        ],
    }
    metrics["stratified"] = {
        "i2t": {k: stratified_metrics(i2t_ranks, v, ks) for k, v in labels.items()},
        "t2i": {k: stratified_metrics(t2i_ranks, v, ks) for k, v in labels.items()},
    }

    per_sample = []
    for i, sid in enumerate(ids):
        row = sim[i].copy()
        pos = row[i]
        row[i] = -np.inf
        per_sample.append(
            {
                "sample_id": sid,
                "split": split,
                "i2t_rank": int(i2t_ranks[i]),
                "t2i_rank": int(t2i_ranks[i]),
                "positive_sim": float(pos),
                "best_wrong_sim": float(row.max()),
                "best_wrong_id": ids[int(row.argmax())],
                "modality": labels["modality_proxy"][i],
                "length_band": labels["caption_length_band"][i],
                "concept_group": labels["concept_group"][i],
            }
        )
    return metrics, per_sample


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/metadata/clean_manifest.jsonl")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--splits", nargs="*", default=["validation", "test"])
    ap.add_argument("--ks", nargs="*", type=int, default=[1, 5, 10])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260701)
    ap.add_argument("--out", default="reports/retrieval_metrics.json")
    ap.add_argument("--per-sample-out", default="reports/per_sample_retrieval.jsonl")
    args = ap.parse_args()

    cfg = RetrievalConfig(ks=tuple(args.ks), n_boot=args.n_boot, seed=args.seed)
    manifest = load_manifest(args.manifest)

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "ks": list(cfg.ks),
            "n_boot": cfg.n_boot,
            "seed": cfg.seed,
            "ambiguous_rule": {
                "caption_containment_ge": cfg.ambiguous_caption_containment,
                "image_phash_hamming_lt": cfg.ambiguous_image_hamming,
            },
        },
        "splits": {},
    }
    all_per_sample: list[dict] = []
    for split in args.splits:
        metrics, per_sample = eval_split(split, args.cache_dir, manifest, cfg)
        out["splits"][split] = metrics
        all_per_sample.extend(per_sample)
        print(
            f"[{split}] n={metrics['n_candidates']} "
            f"i2t R@1={metrics['i2t']['recall@1']:.4f} R@5={metrics['i2t']['recall@5']:.4f} "
            f"R@10={metrics['i2t']['recall@10']:.4f} MRR={metrics['i2t']['mrr']:.4f} | "
            f"t2i R@1={metrics['t2i']['recall@1']:.4f} MRR={metrics['t2i']['mrr']:.4f} | "
            f"AUROC(clean)={metrics['negative_audit']['auroc_clean']:.4f}",
            flush=True,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(args.per_sample_out, "w", encoding="utf-8", newline="\n") as f:
        for row in all_per_sample:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"[saved] {args.out}")
    print(f"[saved] {args.per_sample_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
