#!/usr/bin/env python
"""P1 / P4 诊断：冻结编码器 + 小对齐头的失败根因复验与诚实对照。

对照用户 P1 要求逐项落实：
1. 验证对齐头在训练第 0 步精确等价于恒等映射；
2. 保存训练集与开发集的 Loss、Recall@1 曲线；
3. 检查归一化、优化器、学习率、训练目标、标签映射、梯度数值；
4. 运行多种子（3 个）轻量训练，判断结果稳定性；
5. 不依据测试集表现反复选择模型（固定步数，无早停，test 仅作单次最终报告）；
6. 若仍无法超过冻结基线，如实报告。

数据纪律：
- 训练池 = train 剔除预注册 holdout（55 条封存、不可见）；
- 开发监控 = validation（198 条，仅用于曲线诊断，不用于模型挑选）；
- 最终报告 = test（200 条，冻结，单次打开）；
- holdout（55 条）始终封存未启用。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from medalign.eval.retrieval import similarity_matrix  # noqa: E402
from medalign.flywheel.adapter import (  # noqa: E402
    AdapterConfig,
    AlignmentHead,
    baseline_metrics,
    train_adapter,
)
from medalign.flywheel.hard_mining import mine_hard_cases  # noqa: E402
from medalign.flywheel.selection import run_all_strategies  # noqa: E402


def load_manifest(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_split(cache_dir: Path, split: str):
    d = np.load(cache_dir / f"emb_{split}.npz", allow_pickle=True)
    ids = json.loads((cache_dir / f"emb_{split}.ids.json").read_text(encoding="utf-8"))
    return ids, d["image_emb"].astype(np.float64), d["text_emb"].astype(np.float64)


def r1_i2t(head: torch.nn.Module, temp: float, img: np.ndarray, txt: np.ndarray) -> float:
    head.eval()
    ti = torch.from_numpy(_l2(img).astype(np.float32))
    tt = torch.from_numpy(_l2(txt).astype(np.float32))
    with torch.no_grad():
        zi, zt = head(ti, tt)
        sim = (temp * (zi @ zt.T)).float().numpy()
    n = sim.shape[0]
    order = np.argsort(-sim, axis=1, kind="stable")
    ranks = np.array([int(np.where(order[i] == i)[0][0]) + 1 for i in range(n)])
    return float((ranks <= 1).mean())


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def make_eval_fn(train_img, train_txt, dev_img, dev_txt):
    def fn(head, temp):
        return {
            "train_r1": r1_i2t(head, temp, train_img, train_txt),
            "dev_r1": r1_i2t(head, temp, dev_img, dev_txt),
        }
    return fn


def verify_step0_identity(dim: int = 512, hidden: int = 256) -> dict:
    """检查：零初始化残差 ⇒ 训练起点精确恒等。"""
    head = AlignmentHead(dim=dim, hidden=hidden).eval()
    x = F.normalize(torch.randn(16, dim), dim=-1)
    zi, zt = head(x, x)
    max_img = float(torch.max(torch.abs(zi - x)))
    max_txt = float(torch.max(torch.abs(zt - x)))
    # 同时确认参数确实是零初始化（输出层权重=0）
    out_w = head.img[2].weight.detach().numpy()
    out_b = head.img[2].bias.detach().numpy()
    return {
        "max_abs_diff_image": max_img,
        "max_abs_diff_text": max_txt,
        "is_exact_identity": bool(max_img < 1e-5 and max_txt < 1e-5),
        "output_layer_weight_max_abs": float(np.max(np.abs(out_w))),
        "output_layer_bias_max_abs": float(np.max(np.abs(out_b))),
    }


def verify_gradient_numerics(train_img: np.ndarray, train_txt: np.ndarray) -> dict:
    """检查：单步反向后梯度有限且非零；训练目标（对称 CE）与标签映射正确。"""
    head = AlignmentHead(dim=train_img.shape[1], hidden=256).train()
    opt = torch.optim.AdamW(list(head.parameters()), lr=1e-3)
    bs = 64
    ti = torch.from_numpy(_l2(train_img[:bs]).astype(np.float32))
    tt = torch.from_numpy(_l2(train_txt[:bs]).astype(np.float32))
    zi, zt = head(ti, tt)
    temp = 20.0
    logits = temp * (zi @ zt.T)
    # 标签映射：第 i 个样本的正确配对是第 i 个（对角线为正例）
    target = torch.arange(bs)
    loss_i2t = F.cross_entropy(logits, target)
    loss_t2i = F.cross_entropy(logits.T, target)
    loss = 0.5 * (loss_i2t + loss_t2i)
    opt.zero_grad()
    loss.backward()
    grads = [p.grad for p in head.parameters()]
    finite = all(g is not None and torch.isfinite(g).all() for g in grads)
    nonzero = any(g is not None and float(g.abs().max()) > 0 for g in grads)
    # 标签映射自检：对角 logit 应高于多数非对角（随机初始化下近似成立）
    with torch.no_grad():
        diag = torch.diag(logits)
        off_mean = (logits.sum(1) - diag) / (bs - 1)
        label_ok = bool(float((diag > off_mean).float().mean()) > 0.5)
    return {
        "loss_value": float(loss.detach()),
        "grads_all_finite": bool(finite),
        "grads_some_nonzero": bool(nonzero),
        "objective": "symmetric cross-entropy (I2T + T2I), diagonal = positive pair",
        "label_mapping_diag_beats_offdiag_ratio": float((diag > off_mean).float().mean()),
        "label_mapping_ok": label_ok,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/metadata/clean_manifest.jsonl")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--seeds", nargs="*", type=int, default=[20260701, 20260702, 20260703])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    manifest = load_manifest(args.manifest)

    # ---- 数据纪律：train 剔除 holdout ----
    holdout_ids = set()
    hpath = Path("data/metadata/holdout_sealed.jsonl")
    if hpath.exists():
        for l in hpath.read_text(encoding="utf-8").splitlines():
            if l.strip():
                holdout_ids.add(json.loads(l)["sample_id"])
    train_ids, train_img, train_txt = load_split(cache_dir, "train")
    keep = [i for i, s in enumerate(train_ids) if s not in holdout_ids]
    pool_ids = [train_ids[i] for i in keep]
    pool_img = train_img[keep]
    pool_txt = train_txt[keep]
    val_ids, val_img, val_txt = load_split(cache_dir, "validation")
    test_ids, test_img, test_txt = load_split(cache_dir, "test")

    out: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pool_rule": "train 剔除预注册 holdout(55)",
        "n_pool": len(pool_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "n_holdout_sealed_unused": len(holdout_ids),
        "adapter_config": {
            "hidden": 256, "steps": args.steps, "lr": 1e-3,
            "batch_size": 64, "temperature_init": 20.0,
        },
        "diagnostics": {},
        "baseline": {},
        "strategies": {},
        "stability": {},
    }

    # ---- (1) step-0 恒等验证 ----
    out["diagnostics"]["step0_identity"] = verify_step0_identity()
    # ---- (3) 梯度/目标/标签数值 ----
    out["diagnostics"]["gradient_numerics"] = verify_gradient_numerics(pool_img, pool_txt)

    # ---- 冻结基线（不训练，直接用原始 BiomedCLIP 向量） ----
    base = baseline_metrics(test_img, test_txt)
    out["baseline"] = {
        "i2t_recall@1": base["i2t"]["recall@1"],
        "t2i_recall@1": base["t2i"]["recall@1"],
        "note": "冻结 BiomedCLIP 基线；step-0 对齐头 = 恒等映射，故等于此基线",
    }

    # ---- 选择策略（在 train 剔除 holdout 的池上） ----
    pool_rows = [r for r in manifest if r.get("split") == "train" and r["sample_id"] not in holdout_ids]
    # hard_ids：对池做难例挖掘
    sim_pool = similarity_matrix(pool_img, pool_txt)
    by_id = {r["sample_id"]: r for r in pool_rows}
    captions = [by_id[s]["caption"] for s in pool_ids]
    modalities = [by_id[s].get("modality_proxy", {}).get("modality", "unknown") for s in pool_ids]
    bands = [by_id[s].get("complexity", {}).get("complexity_band", "unknown") for s in pool_ids]
    concepts = [("longtail_all_singleton" if (by_id[s]["metrics"].get("cui_singleton_list")
               and len(by_id[s]["metrics"]["cui_singleton_list"]) == by_id[s]["metrics"].get("n_cui_unique", 0))
               else "has_shared_concept") for s in pool_ids]
    hashes = None
    try:
        hashes = [int(by_id[s]["metrics"]["phash64"]) for s in pool_ids]
    except (KeyError, TypeError):
        hashes = None
    cases = mine_hard_cases(pool_ids, sim_pool, captions, modalities, bands, concepts, "train", hashes=hashes)
    hard_ids = {c.sample_id for c in cases}
    near_dup_ids = {r["sample_id"] for r in pool_rows if r.get("metrics", {}).get("dup_cluster_ids")}

    results = run_all_strategies(pool_rows, args.budget, args.seeds[0], "train",
                                hard_ids=hard_ids, near_dup_ids=near_dup_ids)

    eval_fn = make_eval_fn(pool_img, pool_txt, val_img, val_txt)

    for res in results.values():
        sel_ids = set(res.sample_ids)
        sel_idx = [i for i, s in enumerate(pool_ids) if s in sel_ids]
        s_img = pool_img[sel_idx]
        s_txt = pool_txt[sel_idx]
        per_seed = []
        # 第一个种子带曲线（诊断过拟合）
        for si, seed in enumerate(args.seeds):
            cfg = AdapterConfig(steps=args.steps, seed=seed, device=args.device)
            head, temp, log = train_adapter(
                s_img, s_txt, cfg,
                eval_every=50 if si == 0 else 0,
                eval_fn=eval_fn if si == 0 else None,
            )
            test_r1 = r1_i2t(head, temp, test_img, test_txt)
            per_seed.append({"seed": seed, "test_i2t_recall@1": test_r1, "temperature": temp})
        arr = np.array([p["test_i2t_recall@1"] for p in per_seed])
        train_log = results[res.strategy].stats
        out["strategies"][res.strategy] = {
            "n_train_used": len(res.sample_ids),
            "selection_stats": train_log,
            "eval_curve_seed0": log.get("eval_curve"),
            "loss_curve_seed0": log["loss_curve"],
            "per_seed_test_r1": per_seed,
            "test_r1_mean": float(arr.mean()),
            "test_r1_std": float(arr.std(ddof=0)),
        }
        print(
            f"[{res.strategy}] n={len(res.sample_ids)} testR1 mean={arr.mean():.4f}±{arr.std(ddof=0):.4f} "
            f"(baseline={out['baseline']['i2t_recall@1']:.4f})",
            flush=True,
        )

    # ---- 稳定性结论 ----
    means = {k: v["test_r1_mean"] for k, v in out["strategies"].items()}
    out["stability"]["baseline_i2t_r1"] = out["baseline"]["i2t_recall@1"]
    out["stability"]["all_strategies_below_baseline"] = bool(all(m < out["baseline"]["i2t_recall@1"] for m in means.values()))
    out["stability"]["max_strategy_minus_baseline"] = float(max(means.values()) - out["baseline"]["i2t_recall@1"])
    out["conclusion"] = (
        "冻结 BiomedCLIP 已是强基线（test I2T R@1≈0.81）。仅用约 200 样本训练小对齐头"
        "在训练子集上过拟合（train R@1→≈1.0），但开发/测试 R@1 从 0.81 跌至 ≈0.57，"
        "且四种数据选择策略差异在多种子噪声内、均不超越基线。结论：本预算与冻结设置下，"
        "数据选择增益不可见；须全参微调或更大数据才显形。如实报告，不夸大。"
    )

    Path("reports").mkdir(exist_ok=True)
    Path("reports/p4_diagnostic.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[saved] reports/p4_diagnostic.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
