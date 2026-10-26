"""图像感知哈希：dHash / pHash。

两种都是**可复现、可解释**的经典方法，用来检测近重复图像，
而不是作为"图像是否医学正确"的判断依据。

依赖：Pillow + numpy（纯 nécessaire 数值计算，不需要 scipy/sklearn）。
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image

from .hashing import sha256_bytes


def load_image_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


def _dct_matrix(n: int) -> np.ndarray:
    """DCT-II 矩阵（正交版本），手写避免 scipy 依赖。"""
    x = np.arange(n, dtype=np.float64)
    mat = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        mat[k, :] = np.cos((2 * x + 1) * k * math.pi / (2 * n))
        if k == 0:
            mat[k, :] *= math.sqrt(1.0 / n)
        else:
            mat[k, :] *= math.sqrt(2.0 / n)
    return mat


_DCT8_CACHE: dict[int, np.ndarray] = {}


def _get_dct(n: int) -> np.ndarray:
    if n not in _DCT8_CACHE:
        _DCT8_CACHE[n] = _dct_matrix(n)
    return _DCT8_CACHE[n]


def _resample_nearest():
    return getattr(Image, "Resampling", Image).NEAREST


def _resample_bilinear():
    return getattr(Image, "Resampling", Image).BILINEAR


def _resample_lanczos():
    return getattr(Image, "Resampling", Image).LANCZOS


def dhash(img: Image.Image, hash_size: int = 8) -> int:
    """差值哈希：比较相邻像素亮度，输出 hash_size^2 bit 整数。"""
    g = img.convert("L").resize((hash_size + 1, hash_size), _resample_bilinear())
    arr = np.asarray(g, dtype=np.float64)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(bool(b))
    return bits


def phash(img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """ perceptive hash：DCT 低频 8x8（去掉 DC）的中位数阈值编码。"""
    img_size = hash_size * highfreq_factor
    g = img.convert("L").resize((img_size, img_size), _resample_lanczos())
    arr = np.asarray(g, dtype=np.float64)
    m = _get_dct(img_size)
    coeff = m @ arr @ m.T
    low = coeff[:hash_size, :hash_size].copy()
    low[0, 0] = 0.0  # 去掉 DC 分量，避免整体亮度干扰
    flat = low.flatten()
    # 保留 DC 位（置 0 后自然不具备信息），其余位参与中位数阈值比较
    med = float(np.median(flat))
    bits = 0
    for v in flat:
        bits = (bits << 1) | int(v > med)
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count() if hasattr(int, "bit_count") else bin(a ^ b).count("1")


def hamming_norm(a: int, b: int, bits: int = 64) -> float:
    """归一化汉明距离，0=完全相同，1=完全不同。"""
    return hamming(a, b) / max(1, bits)


def image_stats(img: Image.Image) -> dict:
    arr = np.asarray(img.convert("L"), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "unique_levels": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "unique_levels": int(np.unique(arr.round().astype(np.uint8)).size),
    }


def compute_hashes(data: bytes) -> dict:
    """从原始字节一次性算出 sha256 与两种感知哈希，用于血缘与去重。"""
    img = load_image_bytes(data)
    return {
        "bytes_sha256": sha256_bytes(data),
        "dhash64": dhash(img),
        "phash64": phash(img),
        "width": img.size[0],
        "height": img.size[1],
        "mode": img.mode,
        "format": img.format,
        **{f"gray_{k}": v for k, v in image_stats(img).items()},
    }
