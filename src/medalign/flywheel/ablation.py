"""数据选择策略的消融比较。

对比维度（全部是**数据侧**指标，不含任何"模型变强"的暗示）：
- 选中样本数、模态/解剖部位覆盖、概念(CUI)覆盖；
- 长尾概念占比、图注长度分布；
- 近重复样本占比、困难样本占比；
- 策略两两之间的重叠度。
"""

from __future__ import annotations

from typing import Sequence

from .selection import SelectionResult


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def compare_selections(results: Sequence[SelectionResult]) -> dict:
    table: dict = {}
    for r in results:
        s = dict(r.stats)
        s["strategy"] = r.strategy
        s["budget"] = r.budget
        s["seed"] = r.seed
        table[r.strategy] = s

    overlap: dict = {}
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a, b = results[i], results[j]
            overlap[f"{a.strategy}__vs__{b.strategy}"] = round(
                jaccard(a.sample_ids, b.sample_ids), 4
            )

    return {
        "table": table,
        "pairwise_jaccard": overlap,
        "note": (
            "这些是数据选择策略的**数据侧**描述；"
            "是否带来模型指标差异必须由 P4 的对照训练给出，不能从这里推断。"
        ),
    }
