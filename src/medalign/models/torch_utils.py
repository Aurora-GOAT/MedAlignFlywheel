"""PyTorch 运行环境的统一入口：设备选择、随机种子、指纹收集。

为什么单独抽一层：所有"这次实验到底在什么条件下跑的"信息必须集中采集，
否则报告里的运行环境声明会散落在各个脚本里，最终对不上。
"""

from __future__ import annotations

import platform
import random
import sys
from typing import Any

import numpy as np


def pick_device(prefer_cuda: bool = True) -> str:
    """选择设备。CPU 永远可用，CUDA 不可用时静默降级（并把结果写进指纹）。"""
    import torch

    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_all_seeds(seed: int) -> dict[str, Any]:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return {"python": seed, "numpy": seed, "torch": seed}


def collect_env_fingerprint(device: str) -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "torch_cuda_compiled": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "numpy_version": np.__version__,
        "device_used": device,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device() if device.startswith("cuda") else 0
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        info["gpu_capability"] = list(torch.cuda.get_device_capability(idx))
        info["gpu_total_mem_mb"] = round(torch.cuda.get_device_properties(idx).total_memory / 1e6, 1)
    return info
