"""向量索引：归一化 + 暴力 Top-K 检索 + 落盘。

候选规模在本项目里只有几百到几千条，暴力点积比任何近似索引都更准、
也更容易验证正确性，所以这里不做 ANN。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


@dataclass
class EmbeddingIndex:
    """一组已归一化的向量，支持按另一组向量做 Top-K 检索。"""

    ids: list[str]
    vectors: np.ndarray  # (n, d) 已归一化

    @classmethod
    def build(cls, ids: list[str], vectors: np.ndarray) -> "EmbeddingIndex":
        return cls(ids=list(ids), vectors=l2_normalize(vectors))

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])

    def query(self, queries: np.ndarray, top_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """返回 (indices, scores)，形状均为 (n_query, top_k)。"""
        q = l2_normalize(queries)
        sims = q @ self.vectors.T  # (n_query, n)
        k = min(top_k, self.size)
        idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(sims.shape[0])[:, None]
        scores = sims[rows, idx]
        order = np.argsort(-scores, axis=1)
        idx = idx[rows, order]
        scores = scores[rows, order]
        return idx, scores

    def save(self, path: str | Path, ids_path: str | Path | None = None) -> None:
        np.save(str(path), self.vectors)
        if ids_path:
            Path(ids_path).write_text("\n".join(self.ids), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, ids_path: str | Path) -> "EmbeddingIndex":
        vectors = np.load(str(path))
        ids = Path(ids_path).read_text(encoding="utf-8").splitlines()
        return cls(ids=ids, vectors=vectors)


def index_array(vectors: np.ndarray, ids: list[str] | None = None) -> EmbeddingIndex:
    """从裸矩阵构造索引（ids 缺省时用序号占位）。"""
    n = int(np.asarray(vectors).shape[0])
    return EmbeddingIndex.build(ids or [str(i) for i in range(n)], vectors)
