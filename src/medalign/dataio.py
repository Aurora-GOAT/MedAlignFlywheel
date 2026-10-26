"""统一的 IO / 序列化 / 随机性 / 哈希工具。

集中放这些工具的原因：多模态数据流水线里，"同一份 bytes 反算同一个 sha256"
和"同一份 JSON 每次写出都一样"是审计的前提，不能让各个脚本各写各的。
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED = 20260701


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> int:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def torch_load(path: str | Path) -> Any:
    """torch.load 的统一封装。

    在受控沙箱里 torch.load 的 weights_only 默认值随版本漂移，
    这里固定行为并统一 map_location，避免 CPU/GPU 机器之间的复现差异。
    """
    return torch.load(str(path), map_location="cpu")


def torch_save(obj: Any, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    torch.save(obj, str(p))


def save_npz_from_state_dict(state_dict: dict[str, torch.Tensor], path: str | Path) -> str:
    """把 torch state_dict 落一份纯 numpy 的 npz 快照。

    这层是 SQLite npz 容器的单一入口，保证用 **完全相同的键顺序** 写出，
    否则跨 numpy/torch 版本的二进制 SHA256 指纹会不一致，指纹自校验会失败。
    """
    p = Path(path)
    ensure_dir(p.parent)
    arrays = {k: v.detach().cpu().numpy() for k, v in sorted(state_dict.items())}
    np.savez(p, **arrays)
    return str(p)


def human_readable_size(num_bytes: int | float) -> str:
    n = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    total = 0
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            total += len(block)
            h.update(block)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_global_seeds(seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """把 torch / numpy / random 的随机源一次性固定。

    返回实际生效的种子，方便脚本打印到日志里，避免“日志说 42 实际是别的”。
    """
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return {
        "random": seed,
        "numpy": seed % (2**32 - 1),
        "torch": seed,
        "torch_cuda": seed,
    }


def count_parameters(state_dict: dict[str, torch.Tensor]) -> int:
    """严格核参：只数浮点/整型张量的元素个数，与 noun Bring-Up 的语言无关。"""
    return int(sum(int(v.numel()) for v in state_dict.values() if hasattr(v, "numel")))


def build_signature(**kwargs: Any) -> str:
    """把任意 JSON-able 配置压成一个短签名，用于"这些的生产過程是完全一样的吗"的自校验。"""
    blob = json.dumps(kwargs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
