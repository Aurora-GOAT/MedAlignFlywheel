"""难例挖掘：从真实模型的错误里找值得人工复核的样本。

三类产出
--------
1. **困难正例 (hard_positive)**：配对是对的，但模型把它排得很低。
   这类样本对"图文对齐"最有训练价值，也最可能是图注信息量不足或图像太通用。
2. **风险样本 (risk_false_positive)**：某个错配被模型给出比正确配对更高的相似度。
   **注意**：这不等价于数据错误。可能是：
   (a) 图注本身高度相似（合法）——由 ambiguity 规则排除；
   (b) 图像确实相近（近重复）——同样排除；
   (c) 真的把错配当对了 —— 这才是风险，需人工复核。
3. **错误分布**：按模态 / 图注长度 / 概念组统计，指出模型在哪些子群上系统性出错。

所有输出都带 `review` 空字段，等待人工填写；
未经复核前，任何一条都不能被当成"确认的数据缺陷"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from medalign.common.textnorm import containment, token_set

HARD_MINING_VERSION = "hard-mining-v1.0"


@dataclass
class HardCase:
    sample_id: str
    split: str
    kind: str                 # hard_positive | risk_false_positive
    i2t_rank: int | None
    t2i_rank: int | None
    positive_sim: float
    rival_sim: float | None
    rival_sample_id: str | None
    rival_caption: str | None
    margin: float | None       # positive_sim - rival_sim（越小越糟）
    caption: str
    modality_proxy: str
    length_band: str
    concept_group: str
    ambiguity_excluded: bool = False
    review: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _ambiguity_matrix(
    captions: Sequence[str],
    hashes: Sequence[int] | None,
    containment_thr: float,
    hamming_thr: int,
) -> np.ndarray:
    """标记"不能当作确定错配"的 pair（语义相近或图像近似重复）。"""
    n = len(captions)
    tokens = [token_set(c) for c in captions]
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cont = max(containment(tokens[i], tokens[j]), containment(tokens[j], tokens[i]))
            if cont >= containment_thr:
                mask[i, j] = True
            elif hashes is not None:
                d = bin(int(hashes[i]) ^ int(hashes[j])).count("1")
                if d < hamming_thr:
                    mask[i, j] = True
    return mask


def mine_hard_cases(
    ids: Sequence[str],
    sim: np.ndarray,
    captions: Sequence[str],
    modalities: Sequence[str],
    bands: Sequence[str],
    concepts: Sequence[str],
    split: str,
    hashes: Sequence[int] | None = None,
    hard_rank_threshold: int = 10,
    containment_thr: float = 0.6,
    hamming_thr: int = 10,
) -> list[HardCase]:
    n = len(ids)
    amb = _ambiguity_matrix(captions, hashes, containment_thr, hamming_thr)
    order = np.argsort(-sim, axis=1, kind="stable")
    ranks = np.array([int(np.where(order[i] == i)[0][0]) + 1 for i in range(n)])

    cases: list[HardCase] = []
    for i in range(n):
        pos = float(sim[i, i])
        row = sim[i].copy()
        row[i] = -np.inf
        masked_row = row.copy()
        masked_row[amb[i]] = -np.inf
        j = int(np.argmax(masked_row))
        rival = float(masked_row[j])

        review = {
            "is_true_mismatch": "",      # 人工填：yes / no / unsure
            "reviewer_note": "",         # 人工填
            "confirmed_by": "",          # 人工填
        }

        if ranks[i] > hard_rank_threshold:
            cases.append(
                HardCase(
                    sample_id=ids[i], split=split, kind="hard_positive",
                    i2t_rank=int(ranks[i]), t2i_rank=None,
                    positive_sim=pos, rival_sim=rival if np.isfinite(rival) else None,
                    rival_sample_id=ids[j] if np.isfinite(rival) else None,
                    rival_caption=captions[j] if np.isfinite(rival) else None,
                    margin=pos - rival if np.isfinite(rival) else None,
                    caption=captions[i], modality_proxy=modalities[i],
                    length_band=bands[i], concept_group=concepts[i],
                    ambiguity_excluded=False, review=dict(review),
                )
            )
        if np.isfinite(rival) and rival > pos:
            cases.append(
                HardCase(
                    sample_id=ids[i], split=split, kind="risk_false_positive",
                    i2t_rank=int(ranks[i]), t2i_rank=None,
                    positive_sim=pos, rival_sim=rival,
                    rival_sample_id=ids[j], rival_caption=captions[j],
                    margin=pos - rival,
                    caption=captions[i], modality_proxy=modalities[i],
                    length_band=bands[i], concept_group=concepts[i],
                    ambiguity_excluded=False, review=dict(review),
                )
            )
    return cases


def error_distribution(per_sample: Sequence[dict], key: str) -> dict:
    """按某个维度统计错误率（未进 Top-10 视为失败）与中位排名。"""
    groups: dict[str, list[dict]] = {}
    for row in per_sample:
        groups.setdefault(str(row.get(key)), []).append(row)
    out: dict[str, dict] = {}
    for name, rows in sorted(groups.items()):
        ranks = [r["i2t_rank"] for r in rows if r.get("i2t_rank")]
        fails = [1.0 if r > 10 else 0.0 for r in ranks]
        out[name] = {
            "n": len(rows),
            "recall@10": round(sum(1 - f for f in fails) / max(1, len(fails)), 4),
            "median_rank": float(np.median(ranks)) if ranks else None,
            "mean_positive_sim": round(
                float(np.mean([r["positive_sim"] for r in rows])), 4
            ) if rows else None,
        }
    return out


def summarize_cases(cases: Sequence[HardCase]) -> dict:
    by_kind: dict[str, int] = {}
    by_modality: dict[str, int] = {}
    for c in cases:
        by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
        by_modality[c.modality_proxy] = by_modality.get(c.modality_proxy, 0) + 1
    return {
        "n_cases": len(cases),
        "by_kind": by_kind,
        "by_modality_proxy": dict(sorted(by_modality.items(), key=lambda kv: -kv[1])),
        "version": HARD_MINING_VERSION,
        "review_status": "全部未复核：review 字段为空，不得据此宣称存在数据缺陷",
    }
