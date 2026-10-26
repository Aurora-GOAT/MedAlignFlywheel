"""P4：冻结图文编码器，只训练一个小对齐头（轻量适配）。

为什么这样做
------------
- 目标不是"训练大模型"，而是在**完全相同的训练预算**下，
  比较不同数据选择策略训练出的适配头在**冻结 test** 上的表现。
- 编码器被冻结且向量已缓存，所以训练只在一个 (512→h→512) 的小头上做，
  参数量与步数对所有策略严格一致 —— 这是消融可比性的前提。

能说什么 / 不能说什么
----------------------
- 可以说："在相同预算下，策略 X 训练出的适配头在 test 上 R@1 = ..."。
- 不能说："策略 X 提升了模型能力"，除非差异显著且经过对照；
  本模块输出的是**选择策略的局限与差异**，不是能力证明。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ADAPTER_VERSION = "adapter-v1.0"


@dataclass
class AdapterConfig:
    hidden: int = 256
    steps: int = 400
    lr: float = 1e-3
    batch_size: int = 64
    weight_decay: float = 1e-4
    seed: int = 20260701
    temperature_init: float = 20.0
    device: str = "cpu"


class AlignmentHead(nn.Module):
    """图像侧与文本侧各一个瓶颈 MLP，参数共享结构但不共享权重。"""

    def __init__(self, dim: int = 512, hidden: int = 256):
        super().__init__()
        self.img = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.txt = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        # 初始化为接近恒等，避免训练初期破坏原有对齐
        for seq in (self.img, self.txt):
            nn.init.zeros_(seq[2].weight)
            nn.init.zeros_(seq[2].bias)
            nn.init.xavier_uniform_(seq[0].weight)
            nn.init.zeros_(seq[0].bias)

    def forward(self, img: torch.Tensor, txt: torch.Tensor):
        # 残差结构：mlp 零初始化 ⇒ 前向在训练起点精确等于恒等，
        # 完整保留 BiomedCLIP 原有对齐（基线 R@1≈0.81），训练只学"残差调整"。
        zi = self.img(img) + img
        zt = self.txt(txt) + txt
        return F.normalize(zi, dim=-1), F.normalize(zt, dim=-1)

    @property
    def n_parameters(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def train_adapter(
    train_img: np.ndarray,
    train_txt: np.ndarray,
    cfg: AdapterConfig,
    eval_every: int = 0,
    eval_fn=None,
) -> tuple[AlignmentHead, float, dict]:
    """在给定（子集）训练向量上训练对齐头。返回 (head, temperature, log)。

    eval_every>0 且 eval_fn 可调用时，每 eval_every 步调用 eval_fn(head, temperature)
    一次并记录曲线（用于诊断过拟合，不用于模型挑选）。
    """
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed % (2**32 - 1))

    device = cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"
    head = AlignmentHead(dim=train_img.shape[1], hidden=cfg.hidden).to(device)
    log_temp = torch.tensor(math.log(cfg.temperature_init), device=device, requires_grad=True)

    opt = torch.optim.AdamW(
        list(head.parameters()) + [log_temp], lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    ti = torch.from_numpy(_l2(train_img).astype(np.float32)).to(device)
    tt = torch.from_numpy(_l2(train_txt).astype(np.float32)).to(device)
    n = ti.shape[0]
    bs = min(cfg.batch_size, n)

    head.train()
    losses: list[float] = []
    eval_curve: list[dict] = []
    for step in range(cfg.steps):
        idx = [rng.randrange(n) for _ in range(bs)]
        bi = ti[idx]
        bt = tt[idx]
        zi, zt = head(bi, bt)
        logits = log_temp.exp() * (zi @ zt.T)
        target = torch.arange(bs, device=device)
        loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if eval_every and eval_fn and (step + 1) % eval_every == 0:
            with torch.no_grad():
                rec = eval_fn(head, float(log_temp.exp().detach().cpu()))
            rec = dict(rec)
            rec["step"] = step + 1
            eval_curve.append(rec)

    head.eval()
    log = {
        "n_train": int(n),
        "steps": cfg.steps,
        "batch_size": bs,
        "n_parameters": head.n_parameters,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_mean_last50": float(np.mean(losses[-50:])) if losses else None,
        "device": device,
        "loss_curve": losses,
    }
    if eval_curve:
        log["eval_curve"] = eval_curve
    return head, float(log_temp.exp().detach().cpu()), log


@torch.no_grad()
def evaluate_adapter(
    head: AlignmentHead,
    temperature: float,
    test_img: np.ndarray,
    test_txt: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
    device: str = "cpu",
) -> dict:
    """用训练后的头变换 test 向量，再算检索指标。"""
    head.eval()
    dev = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    head = head.to(dev)
    ti = torch.from_numpy(_l2(test_img).astype(np.float32)).to(dev)
    tt = torch.from_numpy(_l2(test_txt).astype(np.float32)).to(dev)
    zi, zt = head(ti, tt)
    sim = (temperature * (zi @ zt.T)).float().cpu().numpy()

    n = sim.shape[0]
    out: dict = {"n_candidates": n, "i2t": {}, "t2i": {}}
    for name, s in (("i2t", sim), ("t2i", sim.T)):
        order = np.argsort(-s, axis=1, kind="stable")
        ranks = np.array([int(np.where(order[i] == i)[0][0]) + 1 for i in range(n)])
        entry = {f"recall@{k}": float((ranks <= k).mean()) for k in ks}
        entry["mrr"] = float((1.0 / ranks).mean())
        entry["median_rank"] = float(np.median(ranks))
        out[name] = entry
        out[f"{name}_ranks"] = ranks.tolist()
    return out


def baseline_metrics(
    test_img: np.ndarray, test_txt: np.ndarray, ks: tuple[int, ...] = (1, 5, 10)
) -> dict:
    """不训练时的基线（直接用原始 BiomedCLIP 向量）。"""
    sim = _l2(test_img) @ _l2(test_txt).T
    n = sim.shape[0]
    out: dict = {"n_candidates": n, "i2t": {}, "t2i": {}}
    for name, s in (("i2t", sim), ("t2i", sim.T)):
        order = np.argsort(-s, axis=1, kind="stable")
        ranks = np.array([int(np.where(order[i] == i)[0][0]) + 1 for i in range(n)])
        entry = {f"recall@{k}": float((ranks <= k).mean()) for k in ks}
        entry["mrr"] = float((1.0 / ranks).mean())
        out[name] = entry
        out[f"{name}_ranks"] = ranks.tolist()
    return out
