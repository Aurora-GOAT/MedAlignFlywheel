#!/usr/bin/env python
"""P0-3 / P0-4：候选规模（200/500/1000）检索评测 + 同源/重复图像/相似图注泄漏分析。

设计要点
--------
- 候选池规模放大：200 = 仅 test 切分；500 = test+val+部分 train；1000 = test+val+train。
  这检验 80.5% 的 R@1 是否只是"候选太少所以容易"。
- 泄漏分析（P0-4）：在 200 test-only 池内统计近重复图像对（pHash 汉明 <=10）与
  近相同图注对（词面 containment >=0.60）；把查询分成"池内存在混淆项"与"干净"两组，
  对比 R@1，量化检索任务是否被同源/重复/相似样本人为变易。
- 全部只用已冻结 embeddings 与 clean_manifest 的元数据，不做任何参数调整。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from medalign.eval.retrieval import bootstrap_recall, recall_at_k  # noqa: E402
from medalign.common.stats import auroc_rank_based  # noqa: E402

KS = (1, 5, 10)
SEED = 20260701
HAMMING_TH = 10
CONTAIN_TH = 0.60


def _containment(a: set, b: set) -> float:
    if not a:
        return 0.0
    return len(a & b) / len(a)


def _wordset(text: str) -> set:
    return set(t for t in text.lower().split() if t)


def load_emb(split: str, cache_dir: Path):
    d = np.load(cache_dir / f"emb_{split}.npz", allow_pickle=True)
    ids = json.loads((cache_dir / f"emb_{split}.ids.json").read_text(encoding="utf-8"))
    return d["image_emb"].astype(np.float32), d["text_emb"].astype(np.float32), ids


def _block(q_mode: str, c_mode: str, q_ids, c_ids, emb_by_id):
    """返回 [n_q, n_c] 余弦相似度；query 用 q_mode 嵌入，candidate 用 c_mode 嵌入。
    q_mode/c_mode ∈ {'img','txt'}。正确配对位于第 i 行第 i 列（cand_ids[0:n_q]=q_ids）。"""
    q = np.stack([emb_by_id[i][0 if q_mode == "img" else 1] for i in q_ids], axis=0)
    c = np.stack([emb_by_id[j][0 if c_mode == "img" else 1] for j in c_ids], axis=0)
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    cn = c / np.linalg.norm(c, axis=1, keepdims=True)
    return qn @ cn.T


def _ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    n = scores.shape[0]
    ranks = np.empty(n, dtype=np.int64)
    for i in range(n):
        order = np.argsort(-scores[i], kind="stable")
        ranks[i] = int(np.where(order == i)[0][0]) + 1
    return ranks


def _rank_metrics(sim: np.ndarray, n_boot: int, seed: int) -> dict:
    i2t = _ranks_from_scores(sim)  # 行=查询，正确项在列 i
    res = {
        "n_queries": int(sim.shape[0]),
        "n_candidates": int(sim.shape[1]),
        "recall@1": float(recall_at_k(i2t, 1)),
        "recall@5": float(recall_at_k(i2t, 5)),
        "recall@10": float(recall_at_k(i2t, 10)),
        "mrr": float(np.mean(1.0 / i2t)),
        "median_rank": float(np.median(i2t)),
        "recall@1_ci95": bootstrap_recall(i2t, KS, n_boot=n_boot, seed=seed)["recall@1"],
    }
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--manifest", default="data/metadata/clean_manifest.jsonl")
    ap.add_argument("--scales", nargs="*", type=int, default=[200, 500, 1000])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="reports/scale_eval.json")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    splits_data = {sp: load_emb(sp, cache) for sp in ("train", "validation", "test")}

    meta = {}
    for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        meta[r["sample_id"]] = r

    emb_by_id = {}
    for sp in ("train", "validation", "test"):
        img, txt, ids = splits_data[sp]
        for k, sid in enumerate(ids):
            emb_by_id[sid] = (img[k], txt[k])

    scales_out: dict = {"config": {"ks": list(KS), "seed": SEED, "scales": args.scales}}

    # 候选文本池顺序：test -> val -> train
    all_cand_ids: list[str] = []
    for sp in ("test", "validation", "train"):
        all_cand_ids.extend(splits_data[sp][2])
    test_ids = splits_data["test"][2]

    for scale in args.scales:
        cand_ids = all_cand_ids[:scale]
        sim_i2t = _block("img", "txt", test_ids, cand_ids, emb_by_id)  # 行=图查询，列=文候选
        sim_t2i = _block("txt", "img", test_ids, cand_ids, emb_by_id)  # 行=文查询，列=图候选
        m_i2t = _rank_metrics(sim_i2t, args.n_boot, SEED)
        m_t2i = _rank_metrics(sim_t2i, args.n_boot, SEED)
        n = sim_i2t.shape[0]
        pos = sim_i2t[np.arange(n), np.arange(n)]
        flat = sim_i2t.reshape(-1)
        dmask = np.ones(flat.shape[0], dtype=bool)
        dmask[np.arange(n)] = False  # 剔除对角（正例）位置
        off = flat[dmask]
        auroc = float(auroc_rank_based(pos, off))
        scales_out[str(scale)] = {
            "n_queries": m_i2t["n_queries"],
            "n_candidates": m_i2t["n_candidates"],
            "i2t": {k: m_i2t[k] for k in ("recall@1", "recall@5", "recall@10", "mrr", "median_rank", "recall@1_ci95")},
            "t2i": {k: m_t2i[k] for k in ("recall@1", "recall@5", "recall@10", "mrr", "median_rank", "recall@1_ci95")},
            "auroc_all_offdiag": auroc,
        }
        print(
            f"[scale {scale}] I2T R@1={m_i2t['recall@1']:.4f} T2I R@1={m_t2i['recall@1']:.4f} "
            f"AUROC={auroc:.4f}",
            flush=True,
        )

    # ---- 泄漏分析（仅 test-only 200 池）----
    img_emb_t, txt_emb_t, test_ids = splits_data["test"]
    qn = img_emb_t / np.linalg.norm(img_emb_t, axis=1, keepdims=True)
    cn = txt_emb_t / np.linalg.norm(txt_emb_t, axis=1, keepdims=True)
    sim = qn @ cn.T
    n = len(test_ids)
    hashes = [int(meta[i]["metrics"]["phash64"]) for i in test_ids]
    cap_sets = [_wordset(meta[i]["caption"]) for i in test_ids]
    near_dup_img = 0
    near_cap = 0
    confusable = [False] * n
    for i in range(n):
        for j in range(i + 1, n):
            dh = (hashes[i] ^ hashes[j]).bit_count()
            if dh <= HAMMING_TH:
                near_dup_img += 1
                confusable[i] = confusable[j] = True
            ov = max(_containment(cap_sets[i], cap_sets[j]), _containment(cap_sets[j], cap_sets[i]))
            if ov >= CONTAIN_TH:
                near_cap += 1
                confusable[i] = confusable[j] = True
    ranks = _ranks_from_scores(sim)
    r1_full = float(recall_at_k(ranks, 1))
    idx_conf = [i for i in range(n) if confusable[i]]
    idx_clean = [i for i in range(n) if not confusable[i]]
    r1_conf = float(recall_at_k(ranks[idx_conf], 1)) if idx_conf else float("nan")
    r1_clean = float(recall_at_k(ranks[idx_clean], 1)) if idx_clean else float("nan")
    leakage = {
        "pool": "test-only (200)",
        "hamming_threshold": HAMMING_TH,
        "containment_threshold": CONTAIN_TH,
        "n_near_dup_image_pairs": near_dup_img,
        "n_near_identical_caption_pairs": near_cap,
        "n_confusable_queries": len(idx_conf),
        "n_clean_queries": len(idx_clean),
        "r1_full": r1_full,
        "r1_confusable_subset": r1_conf,
        "r1_clean_subset": r1_clean,
        "interpretation": (
            "若 r1_clean 与 r1_full 接近，说明近重复/相似图注对 R@1 的提升很小，"
            "80.5% 主要来自模型本身的跨模态对齐能力，而非任务被人为变易。"
        ),
    }
    print(
        f"[leakage] near_dup_img_pairs={near_dup_img} near_cap_pairs={near_cap} "
        f"confusable={len(idx_conf)} r1_full={r1_full:.4f} r1_clean={r1_clean:.4f}",
        flush=True,
    )

    scales_out["leakage_test_only"] = leakage
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(scales_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
