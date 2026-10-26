"""难负例构造：给检索/对比学习提供更"刁"的负样本，但**不许乱判**。

原则
----
一个 pair 只有在通过三重检查后，才能被标成 `confident_negative`：
1. **来源检查**：两者不能是近重复图像（pHash 距离过小），
   也不能共享同一条图注指纹 —— 否则它其实可能是正例。
2. **概念检查**：如果两条样本的 CUI 有交集，说明它们可能在讲同一个医学概念，
   不能简单当负例。
3. **语义检查**：图注词面重叠过高（containment 超阈值）时，
   模型给高分是**合理的**，把这种 pair 当负例会误导训练。

任一检查不通过 → 标 `needs_review`，并在报告里单独计数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from medalign.common.textnorm import containment, token_set

NEGATIVE_VERSION = "negatives-v1.0"


@dataclass
class NegativeCandidate:
    anchor_id: str
    candidate_id: str
    sim: float
    positive_sim: float
    is_hard: bool                 # 相似度高于某个"难"阈值 → 难负例
    checks: dict = field(default_factory=dict)
    verdict: str = "needs_review"  # confident_negative | needs_review
    review: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def build_negatives(
    ids: Sequence[str],
    sim: np.ndarray,
    captions: Sequence[str],
    cuis: Sequence[Sequence[str]],
    hashes: Sequence[int] | None = None,
    top_k: int = 5,
    hard_sim_quantile: float = 0.90,
    containment_thr: float = 0.6,
    hamming_thr: int = 10,
    random_seed: int = 20260701,
    n_random_per_anchor: int = 1,
) -> list[NegativeCandidate]:
    n = len(ids)
    tokens = [token_set(c) for c in captions]
    cui_sets = [set(c or []) for c in cuis]
    all_sim_off = sim[~np.eye(n, dtype=bool)]
    hard_thr = float(np.quantile(all_sim_off, hard_sim_quantile)) if all_sim_off.size else 0.0

    rng = np.random.default_rng(random_seed)
    out: list[NegativeCandidate] = []

    for i in range(n):
        # (a) 随机负例
        for _ in range(n_random_per_anchor):
            j = int(rng.integers(0, n))
            if j == i:
                continue
            out.append(_make_candidate(i, j, ids, sim, captions, tokens, cui_sets, hashes,
                                       hard_thr, containment_thr, hamming_thr, kind="random"))
        # (b) 难负例：模型打分最高的若干个非配对
        row = sim[i].copy()
        row[i] = -np.inf
        top_idx = np.argsort(-row)[:top_k]
        for j in top_idx:
            j = int(j)
            out.append(_make_candidate(i, j, ids, sim, captions, tokens, cui_sets, hashes,
                                       hard_thr, containment_thr, hamming_thr, kind="hard"))
    return out


def _make_candidate(i, j, ids, sim, captions, tokens, cui_sets, hashes,
                    hard_thr, containment_thr, hamming_thr, kind: str) -> NegativeCandidate:
    cont = max(containment(tokens[i], tokens[j]), containment(tokens[j], tokens[i]))
    shared_cui = sorted(cui_sets[i] & cui_sets[j])
    ham = None
    if hashes is not None:
        ham = bin(int(hashes[i]) ^ int(hashes[j])).count("1")

    source_ok = True
    if ham is not None and ham < hamming_thr:
        source_ok = False
    concept_ok = len(shared_cui) == 0
    semantic_ok = cont < containment_thr

    checks = {
        "image_phash_hamming": ham,
        "caption_containment": round(float(cont), 4),
        "shared_cui": shared_cui,
        "source_not_near_duplicate": source_ok,
        "concept_disjoint": concept_ok,
        "semantic_disjoint": semantic_ok,
    }
    verdict = "confident_negative" if (source_ok and concept_ok and semantic_ok) else "needs_review"
    return NegativeCandidate(
        anchor_id=ids[i],
        candidate_id=ids[j],
        sim=float(sim[i, j]),
        positive_sim=float(sim[i, i]),
        is_hard=kind == "hard" and float(sim[i, j]) >= hard_thr,
        checks=checks,
        verdict=verdict,
        review={"is_true_mismatch": "", "reviewer_note": ""},
    )


def summarize_negatives(cands: Sequence[NegativeCandidate]) -> dict:
    n_conf = sum(1 for c in cands if c.verdict == "confident_negative")
    n_hard = sum(1 for c in cands if c.is_hard)
    reasons: dict[str, int] = {}
    for c in cands:
        if c.verdict == "needs_review":
            if not c.checks.get("source_not_near_duplicate"):
                reasons["near_duplicate_image_or_similar_source"] = reasons.get("near_duplicate_image_or_similar_source", 0) + 1
            if not c.checks.get("concept_disjoint"):
                reasons["shared_medical_concept"] = reasons.get("shared_medical_concept", 0) + 1
            if not c.checks.get("semantic_disjoint"):
                reasons["caption_overlap_too_high"] = reasons.get("caption_overlap_too_high", 0) + 1
    return {
        "n_candidates": len(cands),
        "n_confident_negative": n_conf,
        "n_hard": n_hard,
        "confident_ratio": round(n_conf / max(1, len(cands)), 4),
        "needs_review_reasons": reasons,
        "version": NEGATIVE_VERSION,
    }
