"""通用工具：稳定哈希、种子、JSON/JSONL IO。

本模块刻意只依赖标准库，保证在没有 torch / pillow 的环境下
也能参与数据血缘计算与单元测试。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA_VERSION = "1.0"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    """流式计算文件 SHA256，避免一次性读入大文件。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def short_hash(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def set_global_seed(seed: int) -> None:
    """固定全局随机源；torch 单独处理（见 models/torch_utils）。"""
    import random

    random.seed(seed)


def read_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_json(path: str | os.PathLike, obj: Any, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, sort_keys=False)


def read_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_jsonable(obj: Any) -> Any:
    """把 set / Path / numpy 标量转成可序列化对象。"""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(to_jsonable(v) for v in obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:  # pragma: no cover - 防御性
            return obj
    return obj


def human_mb(num_bytes: float) -> float:
    return round(num_bytes / (1024 * 1024), 3)
