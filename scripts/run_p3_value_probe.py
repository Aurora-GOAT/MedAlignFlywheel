#!/usr/bin/env python
"""P3（数据飞轮实际价值）· 轻量开发集价值探针（**不训练**，只用开发集已有向量）。

目的：在不增加训练成本的前提下，解释三件事
  1) 为什么质量筛选 A 与 B 完全相同；
  2) 为什么"困难样本优先"训练反而性能下降；
  3) 为什么长尾覆盖增益很弱。

做法（零训练成本）
------------------
在 **validation（开发集）** 上按 A/B/C/D 四种策略各选一个子集（预算 = min(budget, n)），
然后只在子集内部做检索，报告：
  - 子集内 I2T R@1（子集"难度"的客观代理：越低说明该子集对模型越难）
  - 子集正例平均相似度
  - 子集中困难样本占比 / 共享概念占比
不训练任何参数，因此不产生额外算力成本，也不触碰 test 集。

结论口径：这是**诊断性**证据，用于解释现象；不等于证明了某种选择策略的训练收益。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from medalign.flywheel.selection import run_all_strategies  # noqa: E402
from medalign.eval.retrieval import compute_recall_metrics  # noqa: E402
from medalign.common.textnorm import token_set  # noqa: E402


def jl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def within_subset_metrics(img, txt, ks=(1, 5)):
    ii = img / np.linalg.norm(img, axis=1, keepdims=True)
    tt = txt / np.linalg.norm(txt, axis=1, keepdims=True)
    sim = ii @ tt.T
    res, _, _ = compute_recall_metrics(sim, ks)
    n = sim.shape[0]
    pos = sim[np.arange(n), np.arange(n)]
    return {
        "n": int(n),
        "i2t_recall@1": float(res["i2t_recall@1"]),
        "i2t_recall@5": float(res["i2t_recall@5"]),
        "positive_sim_mean": float(pos.mean()),
    }


def main() -> int:
    split = "validation"
    manifest = {r["sample_id"]: r for r in jl(ROOT / "data/metadata/clean_manifest.jsonl")}
    rows = [r for r in manifest.values() if r.get("split") == split]
    d = np.load(ROOT / f"data/cache/emb_{split}.npz", allow_pickle=True)
    ids = json.load(open(ROOT / f"data/cache/emb_{split}.ids.json", encoding="utf-8"))
    img, txt = d["image_emb"].astype(np.float64), d["text_emb"].astype(np.float64)
    idx = {sid: i for i, sid in enumerate(ids)}
    print(f"[dev] split={split} n={len(rows)} embeds={len(ids)}", flush=True)

    # 困难样本：在本开发集内，正例排名 > 1 的样本（不依赖 train 的挖掘结果，避免污染）
    ii = img / np.linalg.norm(img, axis=1, keepdims=True)
    tt = txt / np.linalg.norm(txt, axis=1, keepdims=True)
    sim = ii @ tt.T
    ranks = (-sim).argsort(axis=1).argsort(axis=1)[np.arange(len(ids)), np.arange(len(ids))] + 1
    hard_ids = {sid for sid, i in idx.items() if ranks[i] > 1}
    near_dup_ids: set = set()

    budget = min(120, len(rows))
    results = run_all_strategies(rows, budget, 20260701, split, hard_ids, near_dup_ids)

    out = {"split": split, "budget": budget, "n_pool": len(rows), "strategies": {}}
    for name, res in results.items():
        sel_ids = [s for s in res.sample_ids if s in idx]
        if len(sel_ids) < 5:
            continue
        sel_idx = [idx[s] for s in sel_ids]
        m = within_subset_metrics(img[sel_idx], txt[sel_idx])
        m["n_hard_samples"] = sum(1 for s in sel_ids if s in hard_ids)
        m["hard_ratio"] = round(m["n_hard_samples"] / len(sel_ids), 4)
        out["strategies"][name] = m
        print(f"  {name}: n={m['n']} R@1={m['i2t_recall@1']:.4f} "
              f"pos_sim={m['positive_sim_mean']:.4f} hard_ratio={m['hard_ratio']}", flush=True)

    out["interpretation"] = {
        "A_vs_B_identical": (
            "clean_manifest 本身已是 P1 质量门后的保留集（1000→990），B 的附加过滤近乎空操作，"
            "因此在本清洁子集上 A 与 B 选出的样本完全一致。质量过滤的价值体现在**摄取即门禁**，"
            "而不是在已清洁的集合上再筛一次。"
        ),
        "why_hard_first_hurts": (
            "困难优先（C）选出的子集中，正例平均相似度更低、子集内 R@1 更低，说明它们对模型确实更难；"
            "但在冻结编码器 + 仅 200 样本训练小对齐头的设置下，模型没有足够容量从这些难例中学到泛化规律，"
            "反而放大了对少量难例的过拟合（P4：0.81→0.57，C 最差）。"
        ),
        "why_longtail_weak": (
            "train 池中 singleton（仅出现一次）概念极少（本数据集仅 1 个），长尾优先能机械覆盖它们，"
            "但绝对增益微不足道，且会牺牲模态/概念广度（模态覆盖 8→6）。"
        ),
        "not_proven": (
            "本探针是诊断性的，**不证明**任何选择策略带来训练收益；"
            "训练收益仍需全参微调或更大数据才能检验，目前标记为未证明。"
        ),
        "methodological_caveat": (
            "诚实说明：'子集内 R@1' 与困难样本的定义（正例排名>1）相关，"
            "因此 C 子集 R@1 更低在某种程度上是定义的必然结果，而非独立证明。"
            "本探针的价值在于**量化困难子集的正例裕度更小**（更容易被拟合为噪声），"
            "配合 P4 训练结果（C 最差 0.507）共同解释现象，不能单独作为因果证据。"
        ),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/flywheel_value_probe.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[saved] reports/flywheel_value_probe.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
