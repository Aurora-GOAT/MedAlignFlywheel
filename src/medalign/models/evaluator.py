"""Cross-modal retrieval metrics with bootstrap confidence.

All figures computed here come from real model embeddings. The AUROC part is
careful about one thing: in ROCOv2 there are many legit captions that overlap
heavily with each other, so a "negative" pair is only counted as clean after a
verifiable exclusion rule is applied (see ``negative_pairs``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..dataio import PROJECT_ROOT, read_json, write_json

DEFAULT_SEED = 20260701


# ---------------------------------------------------------------------------
# basic numeric helpers
# ---------------------------------------------------------------------------
def l2norm(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def similarity_matrix(features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
    """Cosine similarity, equivalent on normalized vectors to a dot product."""
    return l2norm(features_a) @ l2norm(features_b).T


def _require_square_pairing(sim: np.ndarray) -> None:
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError("rank computation expects a square (n_query, n_item) matrix")


# ---------------------------------------------------------------------------
# ranks
# ---------------------------------------------------------------------------
def _ranks(scores: np.ndarray) -> np.ndarray:
    """Rank (1-based) of the diagonal element in each row, descending."""
    _require_square_pairing(scores)
    # descending order along each row
    order = np.argsort(-scores, axis=1, kind="stable")
    n = scores.shape[0]
    rows = np.arange(n)[:, None]
    ranks = np.argmax(order == rows, axis=1) + 1
    return ranks.astype(np.int64)


def image_to_text_ranks(sim: np.ndarray) -> np.ndarray:
    return _ranks(sim)


def text_to_image_ranks(sim: np.ndarray) -> np.ndarray:
    return _ranks(sim.T)


# ---------------------------------------------------------------------------
# recall / MRR
# ---------------------------------------------------------------------------
def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k))


def mrr(ranks: np.ndarray) -> float:
    return float(np.mean(1.0 / ranks))


def retrieval_metrics(sim: np.ndarray, ks: Sequence[int]) -> dict[str, Any]:
    i2t = image_to_text_ranks(sim)
    t2i = text_to_image_ranks(sim)
    out: dict[str, Any] = {
        "candidate_library_size": int(sim.shape[0]),
        "i2t": {"mrr": mrr(i2t)},
        "t2i": {"mrr": mrr(t2i)},
    }
    for k in ks:
        out["i2t"][f"R@{k}"] = recall_at_k(i2t, k)
        out["t2i"][f"R@{k}"] = recall_at_k(t2i, k)
    return out


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
def bootstrap_recall_interval(
    sim: np.ndarray, k: int, n_boot: int = 10_000, seed: int = DEFAULT_SEED
) -> dict[str, Any]:
    """Percentile bootstrap over queries for i2t / t2i recall@k."""
    rng = np.random.default_rng(seed)
    i2t = image_to_text_ranks(sim)
    t2i = text_to_image_ranks(sim)
    n = len(i2t)
    result: dict[str, Any] = {}
    for name, ranks in (("i2t", i2t), ("t2i", t2i)):
        hits = (ranks <= k).astype(np.float64)
        draws = np.empty(n_boot, dtype=np.float64)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            draws[i] = hits[idx].mean()
        lo, hi = np.percentile(draws, [2.5, 97.5])
        result[name] = {
            "point": float(hits.mean()),
            "lo": float(lo),
            "hi": float(hi),
            "significant": not (lo <= 0.5 <= hi),  # only meaningful when comparing to 0.5
        }
    return result


# ---------------------------------------------------------------------------
# matched / mismatched distributions + AUROC
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NegativeAudit:
    """Accounting for how many off-diagonal pairs were judged too ambiguous to use."""

    n_pairs_total: int
    n_clean: int
    n_ambiguous: int
    fraction_clean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_pairs_total": self.n_pairs_total,
            "n_clean": self.n_clean,
            "n_ambiguous": self.n_ambiguous,
            "fraction_clean": self.fraction_clean,
        }


def negative_pairs(sim: np.ndarray, ambiguity_mask: np.ndarray) -> tuple[np.ndarray, NegativeAudit]:
    """Return the similarities of clean negatives plus how they were selected.

    ``ambiguity_mask[i, j]`` True means pair (i, j) is *not* usable as a clean
    negative (highly overlapping captions / near-duplicate image), so it must be
    excluded from the "definitely mismatched" statistics.
    """
    _require_square_pairing(sim)
    n = sim.shape[0]
    eye = np.eye(n, dtype=bool)
    off = ~eye
    clean_off = off & ~ambiguity_mask
    n_total = int((n * (n - 1)))
    n_clean = int(clean_off.sum())
    audit = NegativeAudit(
        n_pairs_total=n_total,
        n_clean=n_clean,
        n_ambiguous=n_total - n_clean,
        fraction_clean=round(n_clean / max(1, n_total), 4),
    )
    return sim[clean_off], audit


def diagnose_pairs(sim: np.ndarray) -> dict[str, Any]:
    """Distribution of matched vs mismatched similarities plus AUROC."""
    n = sim.shape[0]
    matched = np.diag(sim).astype(np.float64)
    off = sim[~np.eye(n, dtype=bool)].astype(np.float64)
    pos = matched[:, None]
    # AUROC with tie handling, computed as the Mann-Whitney U statistic
    greater = (pos > off).astype(np.float64)
    equal = (pos == off).astype(np.float64)
    auroc = float((greater.sum() + 0.5 * equal.sum()) / max(1, pos.size * off.size))
    return {
        "matched": {
            "mean": float(matched.mean()),
            "std": float(matched.std()),
            "min": float(matched.min()),
            "max": float(matched.max()),
            "median": float(np.median(matched)),
        },
        "mismatched_all": {
            "mean": float(off.mean()),
            "std": float(off.std()),
            "min": float(off.min()),
            "max": float(off.max()),
            "median": float(np.median(off)),
        },
        "auroc_all_pairs": auroc,
    }


# ---------------------------------------------------------------------------
# error analysis by slice
# ---------------------------------------------------------------------------
def metric_by_slice(
    sim: np.ndarray, ks: Sequence[int], labels: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Recall@k and MRR computed per slice, so errors can be attributed."""
    i2t = image_to_text_ranks(sim)
    t2i = text_to_image_ranks(sim)
    out: dict[str, Any] = {}
    for name, mask in labels.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            continue
        entry: dict[str, Any] = {"n": int(mask.sum())}
        for k in ks:
            entry[f"i2t_R@{k}"] = recall_at_k(i2t[mask], k)
            entry[f"t2i_R@{k}"] = recall_at_k(t2i[mask], k)
        entry["i2t_mrr"] = mrr(i2t[mask])
        entry["t2i_mrr"] = mrr(t2i[mask])
        out[str(name)] = entry
    return out


def load_embeddings(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = np.load(npz_path)
    ids = json.loads(Path(str(npz_path).with_suffix(".ids.json")).read_text(encoding="utf-8"))
    return data["image"], data["text"], ids


def _save(report: dict[str, Any], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, report)
    return out_path


mean_reciprocal_rank = mrr
