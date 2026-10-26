"""数据选择策略 A/B/C/D。

**这些都是"数据选择策略"，不是模型能力提升。**
报告里只能说"在相同训练预算下，策略 X 选出的子集在独立 test 上得到什么结果"，
不能说"策略 X 让模型变强"。

- A 原始随机采样
- B 质量过滤后随机
- C 质量过滤 + 困难样本优先
- D 质量过滤 + 长尾概念覆盖优先
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

SELECTION_VERSION = "selection-v1.0"


@dataclass
class SelectionResult:
    strategy: str
    split: str
    budget: int
    seed: int
    sample_ids: list[str]
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _quality_ok(row: dict) -> bool:
    """质量过滤：保留 decision=keep 且没有"影响可用性"的警告。"""
    if row.get("decision") != "keep":
        return False
    blocking = {"CAPTION_HIGH_DIGIT", "CAPTION_REPETITIVE", "IMAGE_LOW_CONTRAST", "IMAGE_ASPECT_SKEW"}
    return not (set(row.get("warn_codes") or []) & blocking)


def _cuis(row: dict) -> list[str]:
    return list(row.get("cui") or [])


def _modality(row: dict) -> str:
    return (row.get("modality_proxy") or {}).get("modality", "unknown")


def _describe(
    rows_by_id: dict, sample_ids: Sequence[str], near_dup_ids: set, hard_ids: set | None = None
) -> dict:
    sel = [rows_by_id[s] for s in sample_ids if s in rows_by_id]
    mod = Counter(_modality(r) for r in sel)
    region = Counter((r.get("region_proxy") or {}).get("region", "unknown") for r in sel)
    cui = Counter()
    for r in sel:
        cui.update(_cuis(r))
    tokens = [r.get("complexity", {}).get("n_tokens", 0) for r in sel]
    dup = sum(1 for s in sample_ids if s in near_dup_ids)
    n_hard = sum(1 for s in sample_ids if hard_ids and s in hard_ids)
    return {
        "n_selected": len(sample_ids),
        "n_distinct_modality": len(mod),
        "modality_counts": dict(mod.most_common()),
        "n_distinct_region": len(region),
        "region_counts": dict(region.most_common(5)),
        "n_distinct_cui": len(cui),
        "top_cui": dict(cui.most_common(5)),
        "caption_tokens_mean": round(sum(tokens) / max(1, len(tokens)), 2),
        "near_dup_ratio": round(dup / max(1, len(sample_ids)), 4),
        "n_hard_samples": n_hard,
    }


def strategy_a_random(rows: Sequence[dict], budget: int, seed: int, split: str) -> SelectionResult:
    rng = random.Random(seed)
    pool = [r["sample_id"] for r in rows]
    rng.shuffle(pool)
    return SelectionResult("A_random", split, budget, seed, sorted(pool[:budget]))


def strategy_b_quality(rows: Sequence[dict], budget: int, seed: int, split: str) -> SelectionResult:
    rng = random.Random(seed)
    pool = [r["sample_id"] for r in rows if _quality_ok(r)]
    rng.shuffle(pool)
    return SelectionResult("B_quality", split, budget, seed, sorted(pool[:budget]))


def strategy_c_quality_hard(
    rows: Sequence[dict],
    budget: int,
    seed: int,
    split: str,
    hard_ids: Sequence[str],
) -> SelectionResult:
    """质量过滤 + 困难样本优先：先取困难集，不足再用质量池补齐。"""
    rng = random.Random(seed)
    quality_pool = [r["sample_id"] for r in rows if _quality_ok(r)]
    hard_set = set(hard_ids) & set(quality_pool)
    hard = sorted(hard_set)
    rest = [s for s in quality_pool if s not in hard_set]
    rng.shuffle(rest)
    chosen = (hard + rest)[:budget]
    return SelectionResult("C_quality_hard", split, budget, seed, sorted(chosen))


def strategy_d_quality_longtail(
    rows: Sequence[dict],
    budget: int,
    seed: int,
    split: str,
) -> SelectionResult:
    """质量过滤 + 长尾概念覆盖优先：稀缺 CUI / 稀缺模态先选。"""
    rng = random.Random(seed)
    quality_rows = [r for r in rows if _quality_ok(r)]
    cui_df = Counter()
    mod_df = Counter()
    for r in quality_rows:
        cui_df.update(set(_cuis(r)))
        mod_df[_modality(r)] += 1

    def rarity(row: dict) -> float:
        cuis = _cuis(row)
        if not cuis:
            return 0.0
        # 平均稀缺度：出现次数越少越稀缺
        return sum(1.0 / max(1, cui_df[c]) for c in cuis) / len(cuis)

    scored = sorted(
        quality_rows,
        key=lambda r: (-rarity(r), -(1.0 / max(1, mod_df[_modality(r)])), r["sample_id"]),
    )
    chosen = [r["sample_id"] for r in scored[:budget]]
    rng.shuffle(chosen)  # 不改变集合，只打乱呈现顺序
    return SelectionResult("D_quality_longtail", split, budget, seed, sorted(chosen))


def run_all_strategies(
    rows: Sequence[dict],
    budget: int,
    seed: int,
    split: str,
    hard_ids: Sequence[str],
    near_dup_ids: set,
) -> dict[str, SelectionResult]:
    rows_by_id = {r["sample_id"]: r for r in rows}
    hard_set = set(hard_ids)
    results = {
        "A_random": strategy_a_random(rows, budget, seed, split),
        "B_quality": strategy_b_quality(rows, budget, seed, split),
        "C_quality_hard": strategy_c_quality_hard(rows, budget, seed, split, hard_ids),
        "D_quality_longtail": strategy_d_quality_longtail(rows, budget, seed, split),
    }
    for res in results.values():
        res.stats = _describe(rows_by_id, res.sample_ids, near_dup_ids, hard_set)
    return results


def compare_before_after(
    all_rows: Sequence[dict],
    results: dict[str, SelectionResult],
    near_dup_ids: set,
) -> dict:
    """筛选前后对比：数量、覆盖、近重复比例。"""
    rows_by_id = {r["sample_id"]: r for r in all_rows}
    before = _describe(rows_by_id, list(rows_by_id.keys()), near_dup_ids)
    after = {k: v.stats for k, v in results.items()}
    return {
        "before_selection": before,
        "after_selection": after,
        "version": SELECTION_VERSION,
    }
