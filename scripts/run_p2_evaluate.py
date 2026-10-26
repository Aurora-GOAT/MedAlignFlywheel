#!/usr/bin/env python
"""P2：真实 BiomedCLIP 的跨模态检索评测。

产出：
- data/cache/emb_<split>.npz + sim_<split>.npy  （向量与相似度矩阵，P3/P4 复用）
- reports/retrieval_metrics.json                （R@1/5/10、MRR、AUROC、Bootstrap CI、分层错误分析）
- reports/per_sample_ranks.jsonl                （逐样本排名，供 P3 难例挖掘）

冻结原则：test 子集的检索指标只在"协议冻结后"计算一次；
本脚本的 --splits 默认只跑 validation + test，且 test 结果不用于任何参数调整。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from medalign.common.textnorm import token_set  # noqa: E402
from medalign.models import biomedclip  # noqa: E402
from medalign.models.preprocessing import BertWordPieceTokenizer, ImageTransform  # noqa: E402
from medalign.models.evaluator import (  # noqa: E402
    bootstrap_recall_interval,
    diagnose_pairs,
    metric_by_slice,
    negative_pairs,
    retrieval_metrics,
    similarity_matrix,
)

PROJECT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT / "data" / "external" / "models" / "biomedclip"
CLEAN = PROJECT / "data" / "metadata" / "clean_manifest.jsonl"
RAW = PROJECT / "data" / "raw"
CACHE = PROJECT / "data" / "cache"
REPORTS = PROJECT / "reports"

# zero-shot 模态校验用的候选句（固定，不参与训练/调参）
MODALITY_PROMPTS = {
    "x-ray": "a chest x-ray radiograph",
    "computed-tomography": "a computed tomography CT scan",
    "magnetic-resonance": "a magnetic resonance MRI image",
    "ultrasound": "an ultrasound sonography image",
    "histopathology": "a histopathology microscopy image",
    "nuclear-medicine": "a PET nuclear medicine scan",
}


def load_clean(split: str) -> list[dict]:
    rows = [json.loads(line) for line in CLEAN.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r.get("split") == split]


def embed_split(bundle, rows: list[dict], batch_image: int = 16) -> dict:
    paths = [str(RAW / r["image_relpath"]) for r in rows]
    texts = [r["caption"] for r in rows]
    img = biomedclip.embed_images(bundle, paths, batch_size=batch_image)
    txt = biomedclip.embed_texts(bundle, texts, batch_size=64)
    return {"image": img, "text": txt, "ids": [r["sample_id"] for r in rows]}


def ambiguity_mask(rows: list[dict], containment_thr: float = 0.6, hamming_thr: int = 10) -> tuple[np.ndarray, dict]:
    """标记"不能当作确定负例"的错配 pair。

    判定依据（可复核）：
    - 图注词集合互相包含率过高（containment）→ 语义可能等价
    - 图像 pHash 汉明距离过小 → 可能是同一张图的不同版本
    """
    n = len(rows)
    tokens = [token_set(r["caption"]) for r in rows]
    hashes = [int(r["metrics"].get("phash64", 0)) for r in rows]
    mask = np.zeros((n, n), dtype=bool)
    n_caption, n_image = 0, 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = tokens[i], tokens[j]
            cont = 0.0
            if a and b:
                cont = max(len(a & b) / len(a), len(a & b) / len(b))
            ham = bin(hashes[i] ^ hashes[j]).count("1")
            if cont >= containment_thr:
                mask[i, j] = True
                n_caption += 1
            elif ham < hamming_thr:
                mask[i, j] = True
                n_image += 1
    return mask, {"n_by_caption_overlap": n_caption, "n_by_image_hash": n_image, "n_pairs": n * (n - 1)}


def slice_labels(rows: list[dict]) -> dict[str, np.ndarray]:
    """分层错误分析的切片标签（全部来自 P1 的代理字段）。"""
    modality = np.array([r.get("modality_proxy", {}).get("modality", "unknown") for r in rows])
    length = np.array([r.get("complexity", {}).get("complexity_band", "?") for r in rows])
    cuis = [r.get("cui", []) for r in rows]
    concept = np.array(["no_cui" if not c else "has_cui" for c in cuis], dtype=object)
    return {
        "by_modality_proxy": modality,
        "by_caption_length": length,
        "by_concept_presence": concept.astype(str),
    }


def zero_shot_modality_check(bundle, rows: list[dict], limit: int = 200) -> dict:
    """独立的正确性校验：用真实图像做 zero-shot 模态分类，与 P1 的图注代理标签比对。

    这不是评测指标，而是"模型有没有被正确搭起来"的行为学证据：
    如果结构或权重错了，这个一致率会掉到随机水平（≈1/6）。
    """
    sub = rows[:limit]
    paths = [str(RAW / r["image_relpath"]) for r in sub]
    candidates = list(MODALITY_PROMPTS.values())
    img_emb = biomedclip.embed_images(bundle, paths, batch_size=16)
    txt_emb = biomedclip.embed_texts(bundle, candidates, batch_size=len(candidates))
    sim = similarity_matrix(img_emb, txt_emb)
    pred = [list(MODALITY_PROMPTS.keys())[int(i)] for i in sim.argmax(axis=1)]
    gold = [r.get("modality_proxy", {}).get("modality", "unknown") for r in sub]
    evaluated = [(p, g) for p, g in zip(pred, gold) if g in MODALITY_PROMPTS]
    if not evaluated:
        return {"status": "no comparable samples"}
    hit = sum(1 for p, g in evaluated if p == g)
    return {
        "n_evaluated": len(evaluated),
        "n_correct": hit,
        "accuracy": round(hit / len(evaluated), 4),
        "random_baseline": round(1 / len(MODALITY_PROMPTS), 4),
        "note": "以 P1 的图注关键词代理标签为参照；该标签本身是代理，故只作量级校验",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["validation", "test"])
    ap.add_argument("--ks", nargs="*", type=int, default=[1, 5, 10])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--batch-image", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="每个 split 只取前 N 条（调试用）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(REPORTS / "retrieval_metrics.json"))
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((MODEL_DIR / "open_clip_config.json").read_text(encoding="utf-8"))
    pre = cfg.get("preprocess_cfg", {})
    transform = ImageTransform(
        image_size=224,
        mean=tuple(pre.get("mean", (0.48145466, 0.4578275, 0.40821073))),
        std=tuple(pre.get("std", (0.26862954, 0.26130258, 0.27577711))),
    )
    tokenizer = BertWordPieceTokenizer(MODEL_DIR / "vocab.txt")

    bundle = biomedclip.load_biomedclip(
        MODEL_DIR, device=args.device, image_transform=transform, tokenizer=tokenizer
    )
    print("[loaded]", bundle.fingerprint["n_parameters"], "params,", bundle.fingerprint["tensor_sha256"][:16])
    print("[device]", bundle.device)

    report: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_fingerprint": bundle.fingerprint,
        "env": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": bundle.device,
        },
        "splits": {},
    }

    per_sample: list[dict] = []
    for split in args.splits:
        rows = load_clean(split)
        if args.limit:
            rows = rows[: args.limit]
        if not rows:
            print(f"[{split}] empty")
            continue
        print(f"[{split}] n={len(rows)} embedding...", flush=True)
        emb = embed_split(bundle, rows, batch_image=args.batch_image)
        np.savez(CACHE / f"emb_{split}.npz", image=emb["image"], text=emb["text"])
        (CACHE / f"emb_{split}.ids.json").write_text(
            json.dumps(emb["ids"], ensure_ascii=False), encoding="utf-8"
        )

        sim = similarity_matrix(emb["image"], emb["text"])
        np.save(CACHE / f"sim_{split}.npy", sim)

        metrics = retrieval_metrics(sim, args.ks)
        diag = diagnose_pairs(sim)

        amb_mask, amb_info = ambiguity_mask(rows)
        clean_neg, audit = negative_pairs(sim, amb_mask)
        auroc_clean = None
        if clean_neg.size:
            pos = np.diag(sim)[:, None]
            greater = (pos > clean_neg).sum()
            equal = (pos == clean_neg).sum()
            auroc_clean = float((greater + 0.5 * equal) / (pos.size * clean_neg.size))

        boot = {
            f"@{k}": bootstrap_recall_interval(sim, k, n_boot=args.n_boot) for k in args.ks
        }
        slices = metric_by_slice(sim, args.ks, slice_labels(rows))

        report["splits"][split] = {
            "n": len(rows),
            **metrics,
            "similarity_diagnosis": diag,
            "negative_audit": {**audit.to_dict(), **amb_info},
            "auroc_clean_negatives": auroc_clean,
            "bootstrap": boot,
            "error_analysis_by_slice": slices,
        }

        # 逐样本排名（P3 用）
        i2t_rank = (sim.argsort(axis=1)[:, ::-1] == np.arange(len(rows))[:, None]).argmax(axis=1) + 1
        t2i_rank = (sim.argsort(axis=0)[::-1, :] == np.arange(len(rows))[None, :]).argmax(axis=0) + 1
        for i, sid in enumerate(emb["ids"]):
            per_sample.append(
                {
                    "sample_id": sid,
                    "split": split,
                    "i2t_rank": int(i2t_rank[i]),
                    "t2i_rank": int(t2i_rank[i]),
                    "positive_sim": float(sim[i, i]),
                    "best_negative_sim_i2t": float(np.max(np.delete(sim[i], i))),
                    "modality_proxy": rows[i].get("modality_proxy", {}).get("modality"),
                    "caption_len_band": rows[i].get("complexity", {}).get("complexity_band"),
                }
            )
        print(
            f"[{split}] I2T R@1={metrics['i2t']['R@1']:.4f} R@5={metrics['i2t']['R@5']:.4f} "
            f"R@10={metrics['i2t']['R@10']:.4f} MRR={metrics['i2t']['mrr']:.4f} | "
            f"T2I R@1={metrics['t2i']['R@1']:.4f} | AUROC(clean)={auroc_clean}",
            flush=True,
        )

    # 模型行为学校验（用 test 之外的样本，避免动用 test）
    train_rows = load_clean("train")
    report["sanity_zero_shot_modality"] = zero_shot_modality_check(bundle, train_rows)
    print("[sanity] zero-shot modality:", report["sanity_zero_shot_modality"])

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(REPORTS / "per_sample_ranks.jsonl", "w", encoding="utf-8") as f:
        for row in per_sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("[saved]", args.out)
    print("[saved]", REPORTS / "per_sample_ranks.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
