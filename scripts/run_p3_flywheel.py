#!/usr/bin/env python
"""P3：数据飞轮 —— 难例挖掘 + 四种数据选择策略。

产出：
- reports/hard_cases.jsonl        （可人工复核的困难/风险样本清单，含空白复核字段）
- reports/hard_negatives.jsonl    （构造出的难负例，带三重冲突检查与"是否确定负例"标记）
- reports/selection_report.json   （A/B/C/D 四策略的数据侧对比）
- data/metadata/selected_<strategy>.json  （各策略选中的样本 id，供 P4 使用）

纪律：选择策略只使用 **train**（必要时 validation）的信息，
test 不参与任何选择或调参。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from medalign.common.hashing import write_json  # noqa: E402
from medalign.eval.retrieval import RetrievalConfig, similarity_matrix  # noqa: E402
from medalign.flywheel.ablation import compare_selections  # noqa: E402
from medalign.flywheel.hard_mining import mine_hard_cases, summarize_cases  # noqa: E402
from medalign.flywheel.negatives import build_negatives, summarize_negatives  # noqa: E402
from medalign.flywheel.selection import run_all_strategies  # noqa: E402


def load_manifest(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def build_split_arrays(split: str, cache_dir: Path, manifest: list[dict]):
    """读取缓存向量 + manifest，构造 mine_hard_cases / build_negatives 所需的数组。"""
    data = np.load(cache_dir / f"emb_{split}.npz", allow_pickle=True)
    ids = json.loads((cache_dir / f"emb_{split}.ids.json").read_text(encoding="utf-8"))
    by_id = {r["sample_id"]: r for r in manifest if r.get("split") == split}
    order = [i for i, sid in enumerate(ids) if sid in by_id]
    ids_f = [ids[i] for i in order]
    img = data["image_emb"][order].astype(np.float64)
    txt = data["text_emb"][order].astype(np.float64)
    sim = similarity_matrix(img, txt)  # 图像→文本余弦，对角线为正例
    captions = [by_id[s]["caption"] for s in ids_f]
    modalities = [by_id[s].get("modality_proxy", {}).get("modality", "unknown") for s in ids_f]
    bands = [by_id[s].get("complexity", {}).get("complexity_band", "unknown") for s in ids_f]
    concepts = [
        (
            "longtail_all_singleton"
            if (
                by_id[s]["metrics"].get("cui_singleton_list")
                and len(by_id[s]["metrics"]["cui_singleton_list"])
                == by_id[s]["metrics"].get("n_cui_unique", 0)
            )
            else "has_shared_concept"
        )
        for s in ids_f
    ]
    cuis = [by_id[s].get("cui") or [] for s in ids_f]
    hashes = None
    try:
        hashes = [int(by_id[s]["metrics"]["phash64"]) for s in ids_f]
    except (KeyError, TypeError):
        hashes = None
    return ids_f, sim, captions, modalities, bands, concepts, cuis, hashes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/metadata/clean_manifest.jsonl")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260701)
    ap.add_argument("--hard-rank-threshold", type=int, default=10)
    ap.add_argument("--mine-splits", nargs="*", default=["train", "validation"])
    ap.add_argument("--top-m-negatives", type=int, default=5)
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    cfg = RetrievalConfig(seed=args.seed)
    cache_dir = Path(args.cache_dir)
    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)

    out: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "budget": args.budget,
        "seed": args.seed,
        "hard_mining": {},
        "hard_negatives": {},
        "selection": {},
        "note": (
            "本结果来自已修正的 BiomedCLIP 前向（ViTBlock 预归一化顺序已修正），"
            "test 冻结集不参与任何挖掘/选择/调参。"
        ),
    }

    # --- 1) 难例挖掘（train / validation） ---
    all_cases = []
    for split in args.mine_splits:
        try:
            ids_f, sim, captions, modalities, bands, concepts, cuis, hashes = build_split_arrays(
                split, cache_dir, manifest
            )
        except FileNotFoundError:
            print(f"[{split}] 无向量缓存，跳过挖掘", flush=True)
            continue
        cases = mine_hard_cases(
            ids_f, sim, captions, modalities, bands, concepts, split,
            hashes=hashes, hard_rank_threshold=args.hard_rank_threshold,
        )
        all_cases.extend(cases)
        out["hard_mining"][split] = summarize_cases(cases)
        print(f"[{split}] hard cases: {len(cases)}", flush=True)

    with open(reports / "hard_cases.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for c in all_cases:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    print(f"[saved] reports/hard_cases.jsonl ({len(all_cases)} 条)", flush=True)

    # --- 2) 难负例构造 ---
    all_negs = []
    for split in args.mine_splits:
        try:
            ids_f, sim, captions, modalities, bands, concepts, cuis, hashes = build_split_arrays(
                split, cache_dir, manifest
            )
        except FileNotFoundError:
            continue
        negs = build_negatives(
            ids_f, sim, captions, cuis, hashes,
            top_k=args.top_m_negatives, random_seed=args.seed,
        )
        all_negs.extend(negs)
        out["hard_negatives"][split] = summarize_negatives(negs)
        print(
            f"[{split}] negatives: {len(negs)} "
            f"(confident {out['hard_negatives'][split]['n_confident_negative']})",
            flush=True,
        )
    with open(reports / "hard_negatives.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for n in all_negs:
            f.write(json.dumps(n.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    print(f"[saved] reports/hard_negatives.jsonl ({len(all_negs)} 条)", flush=True)

    # --- 3) 四种选择策略（只用 train） ---
    train_rows = [r for r in manifest if r.get("split") == "train"]
    hard_ids = {c.sample_id for c in all_cases if c.split == "train"}
    dup_ids = {r["sample_id"] for r in train_rows if r.get("metrics", {}).get("dup_cluster_ids")}

    results = run_all_strategies(
        train_rows, args.budget, args.seed, "train",
        hard_ids=hard_ids, near_dup_ids=dup_ids,
    )
    out["selection"] = compare_selections(list(results.values()))

    meta_dir = Path("data/metadata")
    meta_dir.mkdir(parents=True, exist_ok=True)
    for res in results.values():
        write_json(meta_dir / f"selected_{res.strategy}.json", res.to_dict())
        print(
            f"[select] {res.strategy}: n={len(res.sample_ids)} "
            f"modality={res.stats['n_distinct_modality']} cui={res.stats['n_distinct_cui']} "
            f"hard={res.stats.get('n_hard_samples', 'NA')}",
            flush=True,
        )

    Path(reports / "selection_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[saved] reports/selection_report.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
