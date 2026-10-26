"""统计工具：Bootstrap 置信区间、McNemar 检验、AUROC。

所有区间估计都要求传入显式种子，保证可复现。
这里只实现项目真正会用到的统计，不追求覆盖面；
实现均为纯 Python + 标准库 random，便于单测。
"""

from __future__ import annotations

import math
import random
from typing import Callable, Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 20260701,
    stat: Callable[[list[float]], float] = None,
) -> tuple[float, float, float]:
    """非参数 Bootstrap。返回 (point_estimate, lo, hi)。

    values: 逐样本取值（例如每条 query 是否命中）。
    """
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return (0.0, 0.0, 0.0)
    statistic = stat or (lambda xs: sum(xs) / len(xs))
    point = statistic(vals)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(statistic([vals[i] for i in idx]))
    boots.sort()
    lo = boots[max(0, int(math.floor(alpha / 2 * n_boot)))]
    hi = boots[min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1)]
    return (point, lo, hi)


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 20260701,
) -> tuple[float, float, float]:
    """a - b 的配对 Bootstrap 差值区间（逐样本一一对应）。

    返回 (diff, lo, hi)。
    """
    av = [float(x) for x in a]
    bv = [float(x) for x in b]
    if len(av) != len(bv):
        raise ValueError("paired bootstrap 要求两组样本一一对应")
    n = len(av)
    diffs = [av[i] - bv[i] for i in range(n)]
    point = sum(diffs) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(sum(diffs[i] for i in idx) / n)
    boots.sort()
    lo = boots[max(0, int(math.floor(alpha / 2 * n_boot)))]
    hi = boots[min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1)]
    return (point, lo, hi)


def mean_difference_significant(diff: tuple[float, float, float]) -> bool:
    """区间是否整体落在 0 的一侧（粗粒度显著性判断）。"""
    _, lo, hi = diff
    return lo > 0 or hi < 0


def wilson_interval(success: int, total: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score 区间，用于比例型指标的快速展示。"""
    if total == 0:
        return (0.0, 0.0)
    p = success / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def auroc_rank_based(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> float:
    """AUROC（Mann-Whitney U 形式）。

    pos_scores: 正确配对的相似度；neg_scores: 错配的相似度。
    返回 P(pos > neg) + 0.5 * P(pos == neq)。
    要求数据量适中——本项目候选库规模在数百级，O(n*m) 可接受。
    """
    # 这里不能用 `if not pos_scores`：numpy 数组会被当成布尔序列，
    # 直接触发 "truth value of an array is ambiguous"。
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    # 用 searchsorted 做向量化计数：
    # left  = 负例中严格小于该正例的数量
    # right = 负例中小于等于的数量（差值即相等）
    pos = np.sort(np.asarray(pos_scores, dtype=np.float64))
    neg = np.sort(np.asarray(neg_scores, dtype=np.float64))
    left = np.searchsorted(neg, pos, side="left")
    right = np.searchsorted(neg, pos, side="right")
    # left = 负例中严格小于该正例的数量 → 即 pos > neg 的配对计数
    greater = int(left.sum())
    equal = int((right - left).sum())
    total = len(pos) * len(neg)
    return (greater + 0.5 * equal) / total


def mcnemar_exact(n01: int, n10: int) -> float:
    """精确 McNemar 双侧检验（二项分布）。n01/n10 为两组不一致对计数。"""
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    # 双侧 p = 2 * P(X <= k), X~Bin(n, 0.5)，截断到 1.0
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * cdf)


def recall_at_k_from_ranks(ranks: Sequence[int], k: int) -> float:
    """ranks: 每条 query 的正确项排序（1-based，未命中给大值/0）。"""
    if not ranks:
        return 0.0
    hits = sum(1 for r in ranks if 0 < r <= k)
    return hits / len(ranks)


def mean_reciprocal_rank(ranks: Sequence[int]) -> float:
    vals = [1.0 / r for r in ranks if r and r > 0]
    return sum(vals) / len(vals) if vals else 0.0


def median(values: Sequence[float]) -> float:
    xs = sorted(float(v) for v in values)
    if not xs:
        return float("nan")
    n = len(xs)
    mid = n // 2
    if n % 2:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])
