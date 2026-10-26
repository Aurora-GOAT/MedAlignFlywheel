"""跨模态检索评测：召回、排序、分布、AUROC、分层分析。

**负例构造的可核验性是本模块的核心约束。**
ROCOv2 里存在大量语义高度重叠的合法图注（同一篇论文的多个子图、
描述同一征象的相近表述）。如果机械地把"非原配对"全部当成错误配对，
AUROC 会被系统性高估。因此这里把负例分成三类，并分别报告：

- clean_negative   : 满足可核验排除条件的错配（用于主结果 AUROC）
- ambiguous_pair   : 图注重叠过高 / 图像近似重复的错配，**不计入负例**
- duplicate_leaked : 近重复簇成员之间的错配（P1 已剔除，评测子集应为空）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from ..common.stats import auroc_rank_based, bootstrap_ci, paired_bootstrap_diff


@dataclass
class RetrievalConfig:
    candidate_size: int = 0
    ks: tuple[int, ...] = (1, 5, 10)
    ambiguous_caption_containment: float = 0.60  # 图注重叠达到该值 → 判为模糊 pair
    ambiguous_image_hamming: int = 10            # pHash 距离小于该值 → 判为模糊 pair
    n_boot: int = 1000
    seed: int = 20260701


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def similarity_matrix(image_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
    """返回 [n_image, n_text] 的余弦相似度矩阵（假设一一配对）。"""
    i = l2_normalize(np.asarray(image_emb, dtype=np.float32))
    t = l2_normalize(np.asarray(text_emb, dtype=np.float32))
    return i @ t.T


def _ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """scores: [n_query, n_item]，第 i 行第 i 列为正确项。返回 1-based 排名。"""
    n = scores.shape[0]
    ranks = np.empty(n, dtype=np.int64)
    for i in range(n):
        # 稳定降序排序；正确项获得最好名次（与常见 CLIP 评测一致）
        order = np.argsort(-scores[i], kind="stable")
        pos = int(np.where(order == i)[0][0]) + 1
        ranks[i] = pos
    return ranks


def compute_recall_metrics(sim: np.ndarray, ks: Sequence[int]) -> dict[str, float]:
    """sim = 图像到文本方向的相似度矩阵，要求第 i 个图文互为正解。"""
    i2t_ranks = _ranks_from_scores(sim)
    t2i_ranks = _ranks_from_scores(sim.T)
    out: dict[str, float] = {}
    for k in ks:
        out[f"i2t_recall@{k}"] = float(recall_at_k(i2t_ranks, k))
        out[f"t2i_recall@{k}"] = float(recall_at_k(t2i_ranks, k))
    out["i2t_mrr"] = float((1.0 / i2t_ranks).mean())
    out["t2i_mrr"] = float((1.0 / t2i_ranks).mean())
    out["i2t_median_rank"] = float(np.median(i2t_ranks))
    out["t2i_median_rank"] = float(np.median(t2i_ranks))
    return out, i2t_ranks, t2i_ranks


def bootstrap_recall(
    ranks: np.ndarray, ks: Sequence[int], n_boot: int = 1000, seed: int = 20260701
) -> dict[str, dict[str, float]]:
    """对每个 K 的召回率做样本级 bootstrap。返回 {metric: {point, lo, hi}}。"""
    out = {}
    for k in ks:
        # 注意这里必须是**逐样本的命中指示**，不能是已经求过均值的标量，
        # 否则 bootstrap 会在一个常数上重采样，区间退化成 0。
        hits = (ranks <= k).astype(np.float64)
        pt, lo, hi = bootstrap_ci(hits, n_boot=n_boot, seed=seed)
        out[f"recall@{k}"] = {"point": pt, "lo": lo, "hi": hi}
    mrr_pt, mrr_lo, mrr_hi = bootstrap_ci(1.0 / ranks, n_boot=n_boot, seed=seed)
    out["mrr"] = {"point": mrr_pt, "lo": mrr_lo, "hi": mrr_hi}
    return out


def classify_negatives(
    sim: np.ndarray,
    caption_tokens: Sequence[set],
    image_hashes: Sequence[int] | None = None,
    cfg: RetrievalConfig | None = None,
) -> dict[str, np.ndarray]:
    """把非对角线的错配 pair 分类成 clean / ambiguous。

    返回布尔掩码 [n, n]，True 表示该位置属于对应类别。
    """
    cfg = cfg or RetrievalConfig()
    n = sim.shape[0]
    diag = np.eye(n, dtype=bool)
    off = ~diag

    from ..common.textnorm import containment

    ambiguous = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            hi_far = True
            if image_hashes is not None and len(image_hashes) == n:
                d = (int(image_hashes[i]) ^ int(image_hashes[j])).bit_count()
                if d < cfg.ambiguous_image_hamming:
                    hi_far = False
            overlap = 0.0
            if caption_tokens and len(caption_tokens) == n:
                overlap = max(
                    containment(caption_tokens[i], caption_tokens[j]),
                    containment(caption_tokens[j], caption_tokens[i]),
                )
                if overlap >= cfg.ambiguous_caption_containment:
                    hi_far = False
            if not hi_far:
                ambiguous[i, j] = True

    clean = off & ~ambiguous
    return {"clean_negative_mask": clean, "ambiguous_mask": ambiguous}


def auditritt(sim: np.ndarray, masks: dict[str, np.ndarray], cfg: RetrievalConfig) -> dict:
    """计算正例相似度 vs 两类负例相似度的分布与 AUROC。"""
    n = sim.shape[0]
    pos = sim[np.arange(n), np.arange(n)].astype(np.float64)
    clean_neg = sim[masks["clean_negative_mask"]].astype(np.float64)
    amb_neg = sim[masks["ambiguous_mask"]].astype(np.float64)

    def dist(x: np.ndarray) -> dict[str, float]:
        if x.size == 0:
            return {"n": 0}
        return {
            "n": int(x.size),
            "mean": float(x.mean()),
            "std": float(x.std()),
            "p05": float(np.percentile(x, 5)),
            "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)),
            "min": float(x.min()),
            "max": float(x.max()),
        }

    return {
        "positive": dist(pos),
        "clean_negative": dist(clean_neg),
        "ambiguous_negative": dist(amb_neg),
        "auroc_clean": auroc_rank_based(pos, clean_neg) if clean_neg.size else float("nan"),
        "auroc_all_offdiag": auroc_rank_based(pos, sim[~np.eye(n, dtype=bool)].astype(np.float64)),
        "n_pairs_total": int(n * n),
        "n_positive": int(pos.size),
        "n_clean_negative": int(clean_neg.size),
        "n_ambiguous_pair": int(amb_neg.size),
        "ambiguous_exclusion_rule": {
            "caption_containment_ge": cfg.ambiguous_caption_containment,
            "image_phash_hamming_lt": cfg.ambiguous_image_hamming,
        },
    }


def stratified_metrics(
    ranks: np.ndarray, labels: Sequence[str], ks: Sequence[int] = (1, 5, 10)
) -> dict[str, dict]:
    """按某个维度（模态/文本长度/概念类别）分组统计召回，便于错误分析。"""
    labels = list(labels)
    out: dict[str, dict] = {}
    for lab in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lab]
        if not idx:
            continue
        sub = ranks[idx]
        entry = {"n": len(idx)}
        for k in ks:
            entry[f"recall@{k}"] = float(recall_at_k(sub, k))
        entry["mrr"] = float((1.0 / sub).mean())
        entry["median_rank"] = float(np.median(sub))
        out[lab] = entry
    return out


def image_to_text_ranks(sim: np.ndarray) -> np.ndarray:
    """对每张图，在全部图注里找它自己那条图注的 1-based 排名。"""
    return _ranks_from_scores(sim)


def text_to_image_ranks(sim: np.ndarray) -> np.ndarray:
    """反过来：对每条图注，在全部图像里找它自己那张图的排名。"""
    return _ranks_from_scores(sim.T)


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    if ranks.size == 0:
        return 0.0
    return float(np.mean(ranks <= k))


def mrr(ranks: np.ndarray) -> float:
    if ranks.size == 0:
        return 0.0
    return float(np.mean(1.0 / ranks))


def metric_by_slice(
    sim: np.ndarray,
    ks: Sequence[int],
    labels: Sequence[str],
    direction: str = "i2t",
) -> dict[str, dict]:
    """按切片标签（模态/文本长度/概念组）统计召回，用于错误归因。"""
    ranks = image_to_text_ranks(sim) if direction == "i2t" else text_to_image_ranks(sim)
    labels = list(labels)
    out: dict[str, dict] = {}
    for lab in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lab]
        if not idx:
            continue
        sub = ranks[idx]
        entry: dict[str, float] = {"n": len(idx)}
        for k in ks:
            entry[f"recall@{k}"] = recall_at_k(sub, k)
        entry["mrr"] = mrr(sub)
        entry["median_rank"] = float(np.median(sub))
        out[str(lab)] = entry
    return out
