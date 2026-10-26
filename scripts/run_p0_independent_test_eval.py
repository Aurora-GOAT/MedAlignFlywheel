#!/usr/bin/env python
"""P0（严格独立 Holdout）· 新独立测试集的**一次性最终评价**。

纪律（关键）
----------
1. **先冻结、后评价**：本脚本第一步就把实验配置（候选规模、指标、种子、权重 SHA256、
   清单 SHA256、近重复阈值）写入 configs/frozen_final_eval.json 并落盘；
   配置冻结之后才计算任何指标，且**只评价一次**（结果文件已存在时拒绝重算，除非 --force）。
2. 测试集来自**从未参与任何开发环节**的公开 ROCOv2 test 行（按 image_id 与图像 SHA256 双重去重，
   排除与既有 1000 条开发样本重复者）。此前封存的 55 条已被审计为受污染（14/55 与 hard_cases 重叠），
   不再作为 holdout 使用。

输出：reports/independent_test_eval.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medalign.common.stats import auroc_rank_based, bootstrap_ci  # noqa: E402
from medalign.common.textnorm import containment, token_set  # noqa: E402
from medalign.common.perceptual import compute_hashes, hamming  # noqa: E402
from medalign.eval.retrieval import bootstrap_recall, compute_recall_metrics  # noqa: E402


def jl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------- 1) 冻结配置（必须在计算任何指标之前） ----------------
def freeze_config(args) -> dict:
    manifest = Path(args.manifest)
    weights = Path(args.weights_dir) / "open_clip_pytorch_model.bin"
    frozen = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "最终独立测试集评价（只做一次）",
        "candidate_scales": args.scales,
        "metrics": ["I2T/T2I R@1", "R@5", "R@10", "MRR", "AUROC(all off-diagonal)"],
        "ks": args.ks,
        "seed": args.seed,
        "n_boot": args.n_boot,
        "model": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "weights_sha256": sha256_file(weights) if weights.exists() else None,
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest) if manifest.exists() else None,
        "manifest_n_rows": len(jl(manifest)) if manifest.exists() else 0,
        "near_dup_thresholds": {
            "image_hamming_max": args.image_hamming,
            "caption_containment_min": args.caption_containment,
        },
        "candidate_ordering": "按 sample_id 字典序取前缀（确定性）",
        "declared_before_evaluation": True,
    }
    out = ROOT / "configs/frozen_final_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[frozen] config -> {out}", flush=True)
    return frozen


# ---------------- 2) 提取向量 ----------------
def embed(rows, args, device):
    import torch
    from PIL import Image
    from medalign.models import biomedclip
    from medalign.models.preprocessing import BertWordPieceTokenizer, ImageTransform

    cfg = json.loads((Path(args.weights_dir) / "open_clip_config.json").read_text(encoding="utf-8"))
    pre = cfg.get("preprocess_cfg", {})
    tf = ImageTransform(224,
                        mean=tuple(pre.get("mean", (0.48145466, 0.4578275, 0.40821073))),
                        std=tuple(pre.get("std", (0.26862954, 0.26130258, 0.27577711))))
    tok = BertWordPieceTokenizer(Path(args.weights_dir) / "vocab.txt")
    bundle = biomedclip.load_biomedclip(args.weights_dir, device=device)
    model = bundle.model

    n = len(rows)
    img_embs = np.zeros((n, 512), dtype=np.float32)
    txt_embs = np.zeros((n, 512), dtype=np.float32)
    failed = []
    bi = args.batch_image
    bt = args.batch_text
    # 图像
    for s in range(0, n, bi):
        chunk = rows[s:s + bi]
        tensors, idxs = [], []
        for j, r in enumerate(chunk):
            # 注意：manifest 里的 image_relpath 是**相对 data/raw** 的路径
            p = ROOT / "data/raw" / r["image_relpath"]
            try:
                im = Image.open(p).convert("RGB")
                tensors.append(torch.from_numpy(tf(im)))
                idxs.append(s + j)
            except Exception as e:
                failed.append({"sample_id": r["sample_id"], "error": str(e)})
        if not tensors:
            continue
        with torch.no_grad():
            e = model.encode_image(torch.stack(tensors).to(device)).float().cpu().numpy()
        for k, gi in enumerate(idxs):
            img_embs[gi] = e[k]
        if s % (bi * 10) == 0:
            print(f"    image {s}/{n}", flush=True)
    # 文本
    for s in range(0, n, bt):
        chunk = rows[s:s + bt]
        ids, mask = tok.batch_encode([r.get("caption") or "" for r in chunk], 256)
        with torch.no_grad():
            e = model.encode_text(torch.from_numpy(ids).to(device),
                                  torch.from_numpy(mask).to(device)).float().cpu().numpy()
        txt_embs[s:s + len(chunk)] = e
    return img_embs, txt_embs, failed, bundle.fingerprint


# ---------------- 3) 近重复分组 ----------------
def near_dup_analysis(rows, args, dev_rows):
    phash_list = []
    for r in rows:
        p = ROOT / "data/raw" / r["image_relpath"]
        try:
            phash_list.append(compute_hashes(p.read_bytes()).get("phash64"))
        except Exception:
            phash_list.append(None)
    toks = [token_set(r.get("caption") or "") for r in rows]

    # 组内近重复
    pairs_img = 0
    pairs_cap = 0
    confusable = set()
    n = len(rows)
    for i in range(n):
        for j in range(i + 1, n):
            same_img = False
            if phash_list[i] is not None and phash_list[j] is not None:
                same_img = hamming(int(phash_list[i]), int(phash_list[j])) <= args.image_hamming
            cap = containment(toks[i], toks[j])
            same_cap = max(cap, containment(toks[j], toks[i])) >= args.caption_containment
            if same_img:
                pairs_img += 1
            if same_cap:
                pairs_cap += 1
            if same_img or same_cap:
                confusable.add(i)
                confusable.add(j)

    # 与开发集（既有 990）之间的近重复
    dev_phash = []
    for r in dev_rows[: args.dev_check_n]:
        p = ROOT / "data/raw/images" / f"{r['sample_id']}.jpg"
        try:
            dev_phash.append(compute_hashes(p.read_bytes()).get("phash64"))
        except Exception:
            dev_phash.append(None)
    cross = 0
    for h in phash_list:
        if h is None:
            continue
        for dh in dev_phash:
            if dh is None:
                continue
            if hamming(int(h), int(dh)) <= args.image_hamming:
                cross += 1
                break
    return {
        "n_in_pool": n,
        "n_near_dup_image_pairs": pairs_img,
        "n_near_identical_caption_pairs": pairs_cap,
        "n_confusable_samples": len(confusable),
        "n_cross_near_dup_with_dev": cross,
        "dev_checked": args.dev_check_n,
        "thresholds": {"image_hamming_max": args.image_hamming,
                       "caption_containment_min": args.caption_containment},
    }


# ---------------- 4) 评价 ----------------
def evaluate(ids_sorted, img, txt, ks, n_boot, seed):
    ii = img / np.linalg.norm(img, axis=1, keepdims=True)
    tt = txt / np.linalg.norm(txt, axis=1, keepdims=True)
    sim = ii @ tt.T
    res, i2t_ranks, t2i_ranks = compute_recall_metrics(sim, ks)
    boot_i = bootstrap_recall(i2t_ranks, ks, n_boot=n_boot, seed=seed)
    boot_t = bootstrap_recall(t2i_ranks, ks, n_boot=n_boot, seed=seed)
    n = sim.shape[0]
    pos = sim[np.arange(n), np.arange(n)]
    off = sim[~np.eye(n, dtype=bool)]
    auroc = float(auroc_rank_based(pos, off))
    out = {
        "n_candidates": int(n),
        "i2t": {f"recall@{k}": float(res[f"i2t_recall@{k}"]) for k in ks},
        "t2i": {f"recall@{k}": float(res[f"t2i_recall@{k}"]) for k in ks},
        "i2t_mrr": float(res["i2t_mrr"]),
        "t2i_mrr": float(res["t2i_mrr"]),
        "i2t_ci95": {f"recall@{k}": boot_i[f"recall@{k}"] for k in ks},
        "t2i_ci95": {f"recall@{k}": boot_t[f"recall@{k}"] for k in ks},
        "auroc_all_offdiag": auroc,
        "positive_sim_mean": float(pos.mean()),
        "negative_sim_mean": float(off.mean()),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/metadata/holdout_v2_manifest.jsonl")
    ap.add_argument("--weights-dir", default="data/external/models/biomedclip")
    ap.add_argument("--scales", nargs="*", type=int, default=[200, 500, 1000])
    ap.add_argument("--ks", nargs="*", type=int, default=[1, 5, 10])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260701)
    ap.add_argument("--batch-image", type=int, default=32)
    ap.add_argument("--batch-text", type=int, default=64)
    ap.add_argument("--image-hamming", type=int, default=8)
    ap.add_argument("--caption-containment", type=float, default=0.80)
    ap.add_argument("--dev-check-n", type=int, default=990)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="reports/independent_test_eval.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_path = ROOT / args.out
    if out_path.exists() and not args.force:
        print(f"[refuse] {out_path} 已存在：配置冻结后只评价一次。如需重算请显式 --force。")
        return 1

    frozen = freeze_config(args)

    rows = jl(ROOT / args.manifest)
    rows = sorted(rows, key=lambda r: r["sample_id"])
    print(f"[data] n={len(rows)} independent samples", flush=True)
    if not rows:
        print("[error] empty manifest; run run_p0_fetch_independent_testset.py first")
        return 1

    import torch
    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[device] {device}", flush=True)

    cache = ROOT / "data/cache/emb_holdout_v2.npz"
    cache_ids = ROOT / "data/cache/emb_holdout_v2.ids.json"
    if cache.exists() and cache_ids.exists():
        d = np.load(cache, allow_pickle=True)
        ids_cached = json.load(open(cache_ids, encoding="utf-8"))
        if ids_cached == [r["sample_id"] for r in rows]:
            img, txt = d["image_emb"], d["text_emb"]
            print("[cache] reuse embeddings", flush=True)
        else:
            img, txt = None, None
    else:
        img, txt = None, None
    if img is None:
        img, txt, failed, fp = embed(rows, args, device)
        # 安全护栏：图像读取失败率过高说明路径/数据有问题，绝不能带着全零向量继续评价
        fail_rate = len(failed) / max(1, len(rows))
        if fail_rate > 0.05:
            print(f"[ABORT] 图像读取失败率 {fail_rate:.1%}（{len(failed)}/{len(rows)}）过高；"
                  f"不保存向量、不产出指标。请检查 image_relpath 与数据目录。", flush=True)
            print(f"[ABORT] 首个失败样例：{failed[0] if failed else None}", flush=True)
            return 1
        np.savez(cache, image_emb=img, text_emb=txt)
        json.dump([r["sample_id"] for r in rows], open(cache_ids, "w", encoding="utf-8"))
        print(f"[saved] embeddings {cache} (failed={len(failed)})", flush=True)

    # 近重复 / 限制分析
    dev_rows = jl(ROOT / "data/metadata/clean_manifest.jsonl")
    nd = near_dup_analysis(rows, args, dev_rows)
    print(f"[near-dup] img_pairs={nd['n_near_dup_image_pairs']} "
          f"cap_pairs={nd['n_near_identical_caption_pairs']} "
          f"confusable={nd['n_confusable_samples']} cross_dev={nd['n_cross_near_dup_with_dev']}", flush=True)

    scales = {}
    for s in args.scales:
        k = min(s, len(rows))
        if k < 2:
            continue
        scales[str(s)] = evaluate(np.arange(k), img[:k], txt[:k], args.ks, args.n_boot, args.seed)
        print(f"[scale {s}] I2T R@1={scales[str(s)]['i2t']['recall@1']:.4f} "
              f"T2I R@1={scales[str(s)]['t2i']['recall@1']:.4f} "
              f"AUROC={scales[str(s)]['auroc_all_offdiag']:.4f}", flush=True)

    result = {
        "frozen_config": frozen,
        "independent_test": {
            "source": "ROCOv2 (eltorio/ROCOv2-radiology) test split, 未经本项目任何开发环节使用",
            "n_samples": len(rows),
            "dedup_rule": "排除与既有 1000 条开发样本相同的 image_id 与图像 SHA256",
        },
        "scales": scales,
        "near_duplicate_and_leakage": nd,
        "limitations": {
            "source_overlap": (
                "ROCOv2 公开接口不暴露来源论文 ID，无法按论文分组；只能按 pHash/图注近重复簇分组。"
                "同一论文的多张子图可能共享高度相似图注，使检索任务偏易（已用近重复统计量化）。"
            ),
            "multi_pair_correctness": (
                "图文配对并非严格一对一：近相同图注的多个图像在语义上都可能是'正确'答案，"
                "因此 R@1 会把语义等价的候选计为错误，属于指标口径的固有保守偏差。"
            ),
            "pretraining_overlap": (
                "BiomedCLIP 在 PMC-15M 上预训练，而 ROCOv2 图像来自 PMC Open Access 子集，"
                "两者存在**实质重叠可能**；因此本测试集上的高 R@1 不能排除部分来自预训练记忆，"
                "不能解读为'模型零样本泛化能力'的纯净证据。"
            ),
            "single_final_evaluation": "配置已冻结且只评价一次；重算需显式 --force 并记录原因。",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
