"""BiomedCLIP（ViT-B/16 + PubMedBERT）的**自包含** torch 实现与加载器。

为什么不用 open_clip / timm / transformers
------------------------------------------
本机 torch 锁在 1.13.1+cu116（GTX 1660 Ti 驱动 511.79，CUDA 上限 11.6），
而 `import open_clip` 在这套组合下会 **segfault**（即使只 import open_clip.tokenizer），
transformers/timm 的新版本又要求 torch>=2。因此这里按官方
`open_clip_config.json` 写明的结构，用 torch 原生模块把网络搭出来。

正确性靠三道闸（不是靠"看起来能跑"）
--------------------------------------
1. **零缺失零多余**：`load_state_dict(strict=True)` + 加载前逐 key 形状比对，
   少一个、多一个、形状不符都直接抛错，绝不部分加载。
2. **指纹**：权重文件 SHA256 + 张量级 SHA256 + 参数量，写进报告。
3. **行为校验**：scripts/run_p2_evaluate.py 会在真实 ROCOv2 上算检索召回
   与 zero-shot 模态分类；如果结构搭错了，这些数字会退化到随机水平。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OFFICIAL_HF_ID = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
ARCH_VERSION = "medalign-biomedclip-torch-v1.0"


@dataclass(frozen=True)
class VisionCfg:
    image_size: int = 224
    patch_size: int = 16
    width: int = 768
    layers: int = 12
    heads: int = 12
    mlp_ratio: int = 4


@dataclass(frozen=True)
class TextCfg:
    vocab_size: int = 30522
    hidden: int = 768
    layers: int = 12
    heads: int = 12
    max_position: int = 512      # checkpoint 原样（与权重一致，保证严格加载）
    context_length: int = 256    # 官方配置：实际推理只用前 256 个位置
    intermediate: int = 3072
    proj_hidden: int = 640       # checkpoint: text.proj.0 -> (640, 768)


EMBED_DIM = 512


# ---------------------------------------------------------------------------
# 视觉塔：timm vit_base_patch16_224（pre-norm / norm_first=True）
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, cfg: VisionCfg):
        super().__init__()
        self.proj = nn.Conv2d(3, cfg.width, kernel_size=cfg.patch_size, stride=cfg.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class ViTAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class ViTBlock(nn.Module):
    """timm Block：post-attention 的 MLP 在 checkpoint 里叫 fc1 / fc2，
    所以这里不用 nn.Sequential（会把参数名落成 0 / 2，加载门禁会报错）。
    """

    def __init__(self, cfg: VisionCfg, eps: float = 1e-6):
        super().__init__()
        dim = cfg.width
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = ViTAttention(dim, cfg.heads)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, dim * cfg.mlp_ratio)
        self.mlp.fc2 = nn.Linear(dim * cfg.mlp_ratio, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp.fc2(F.gelu(self.mlp.fc1(self.norm2(x))))
        return x


class VisionTrunk(nn.Module):
    """对应 checkpoint 的 `visual.trunk.*`。"""

    def __init__(self, cfg: VisionCfg, eps: float = 1e-6):
        super().__init__()
        grid = cfg.image_size // cfg.patch_size
        self.num_patches = grid * grid
        self.patch_embed = PatchEmbed(cfg)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.width))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, cfg.width))
        self.blocks = nn.ModuleList([ViTBlock(cfg, eps=eps) for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(cfg.width, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # class token


class VisionHead(nn.Module):
    """对应 checkpoint 的 `visual.head.*`（Linear(768 -> 512)，无 bias）。"""

    def __init__(self, width: int = 768, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.proj = nn.Linear(width, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class VisionTower(nn.Module):
    def __init__(self, cfg: VisionCfg):
        super().__init__()
        self.trunk = VisionTrunk(cfg)
        self.head = VisionHead(cfg.width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


# ---------------------------------------------------------------------------
# 文本塔：HF BertModel（post-norm）+ open_clip 的 MLP proj
# ---------------------------------------------------------------------------
class BertEmbeddings(nn.Module):
    def __init__(self, cfg: TextCfg, eps: float = 1e-12):
        super().__init__()
        self.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.position_embeddings = nn.Embedding(cfg.max_position, cfg.hidden)
        self.token_type_embeddings = nn.Embedding(2, cfg.hidden)
        self.LayerNorm = nn.LayerNorm(cfg.hidden, eps=eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        tok_type = torch.zeros_like(input_ids)
        x = self.word_embeddings(input_ids)
        x = x + self.position_embeddings(pos)
        x = x + self.token_type_embeddings(tok_type)
        return self.LayerNorm(x)


class BertSelfAttention(nn.Module):
    def __init__(self, cfg: TextCfg):
        super().__init__()
        self.heads = cfg.heads
        self.head_dim = cfg.hidden // cfg.heads
        self.query = nn.Linear(cfg.hidden, cfg.hidden)
        self.key = nn.Linear(cfg.hidden, cfg.hidden)
        self.value = nn.Linear(cfg.hidden, cfg.hidden)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q = self.query(x).reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(x).reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(x).reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5) + attn_mask
        ctx = torch.softmax(scores, dim=-1) @ v
        return ctx.transpose(1, 2).reshape(b, n, c)


class BertAttention(nn.Module):
    def __init__(self, cfg: TextCfg, eps: float = 1e-12):
        super().__init__()
        self.self = BertSelfAttention(cfg)
        self.output = nn.Module()
        self.output.dense = nn.Linear(cfg.hidden, cfg.hidden)
        self.output.LayerNorm = nn.LayerNorm(cfg.hidden, eps=eps)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # BertSelfOutput: LayerNorm(hidden + dense(attn))
        return self.output.LayerNorm(x + self.output.dense(self.self(x, attn_mask)))


class BertLayer(nn.Module):
    def __init__(self, cfg: TextCfg, eps: float = 1e-12):
        super().__init__()
        self.attention = BertAttention(cfg, eps=eps)
        self.intermediate = nn.Module()
        self.intermediate.dense = nn.Linear(cfg.hidden, cfg.intermediate)
        self.output = nn.Module()
        self.output.dense = nn.Linear(cfg.intermediate, cfg.hidden)
        self.output.LayerNorm = nn.LayerNorm(cfg.hidden, eps=eps)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(x, attn_mask)
        ff = F.gelu(self.intermediate.dense(attn_out))
        return self.output.LayerNorm(attn_out + self.output.dense(ff))


class BertEncoder(nn.Module):
    def __init__(self, cfg: TextCfg, eps: float = 1e-12):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(cfg, eps=eps) for _ in range(cfg.layers)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layer:
            x = layer(x, attn_mask)
        return x


class TextTransformer(nn.Module):
    """对应 checkpoint 的 `text.transformer.*`。

    按官方配置 hf_pooler_type="cls_last_hidden_state_pooler"，
    池化就是取最后一层 hidden state 的 CLS 位（权重里没有额外 pooler 参数，
    这与 checkpoint 的实际内容一致）。
    """

    def __init__(self, cfg: TextCfg):
        super().__init__()
        self.embeddings = BertEmbeddings(cfg)
        self.encoder = BertEncoder(cfg)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(input_ids)
        mask = attention_mask[:, None, None, :].to(dtype=x.dtype)
        mask = (1.0 - mask) * torch.finfo(x.dtype).min
        return self.encoder(x, mask)


class TextProj(nn.Module):
    """对应 checkpoint 的 `text.proj.*`：Linear(768->640) -> GELU -> Linear(640->512)，均无 bias。"""

    def __init__(self, cfg: TextCfg, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.proj_hidden, bias=False),
            nn.GELU(),
            nn.Linear(cfg.proj_hidden, embed_dim, bias=False),
        )
        # 让 state_dict 的键保持官方命名 text.proj.0.* / text.proj.2.*
        self.seq[0]._external_name = "0"
        self.seq[2]._external_name = "2"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class TextTower(nn.Module):
    def __init__(self, cfg: TextCfg, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.transformer = TextTransformer(cfg)
        self.proj = TextProj(cfg, embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.transformer(input_ids, attention_mask)
        return self.proj(hidden[:, 0])


class BiomedCLIP(nn.Module):
    def __init__(self, vision_cfg: VisionCfg | None = None, text_cfg: TextCfg | None = None):
        super().__init__()
        self.vision_cfg = vision_cfg or VisionCfg()
        self.text_cfg = text_cfg or TextCfg()
        self.visual = VisionTower(self.vision_cfg)
        self.text = TextTower(self.text_cfg)
        self.logit_scale = nn.Parameter(torch.zeros([]))

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.visual(pixel_values)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.text(input_ids, attention_mask)

    def forward(self, pixel_values, input_ids, attention_mask):
        return self.encode_image(pixel_values), self.encode_text(input_ids, attention_mask)


# ---------------------------------------------------------------------------
# 加载：严格门禁 + 指纹
# ---------------------------------------------------------------------------
class MissingWeightsError(RuntimeError):
    pass


class UnexpectedWeightsError(RuntimeError):
    pass


def _seq_name_to_official(name: str) -> str:
    return name


def load_missing_unexpected(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Bind weights with zero tolerance for missing or unexpected keys."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = list(missing)
    unexpected = list(unexpected)
    if missing:
        raise MissingWeightsError(
            f"{len(missing)} weights missing from the checkpoint: {missing[:8]}"
            + (" ..." if len(missing) > 8 else "")
        )
    if unexpected:
        raise UnexpectedWeightsError(
            f"{len(unexpected)} checkpoint weights do not fit the model: {unexpected[:8]}"
            + (" ..." if len(unexpected) > 8 else "")
        )
    # 真正绑定（strict 模式），确保形状也一致
    model.load_state_dict(state_dict, strict=True)


# HF BertModel 里用来查位置 id 的**常量 buffer**（形状 [1, 512]），不是可学习参数。
# 本实现直接用 torch.arange 生成位置 id，功能等价，所以加载时显式剔除它；
# 剔除项会写进指纹，保证审计可追溯（不是被悄悄丢掉）。
NON_PARAMETER_BUFFERS = ("text.transformer.embeddings.position_ids",)


def _checkpoint_state_dict(weight_path: str | Path) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Read the checkpoint; returns (params, dropped_non_parameter_buffers)."""
    path = Path(weight_path)
    if not path.exists():
        raise FileNotFoundError(f"weights not found: {path}")
    ckpt = torch.load(str(path), map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    params = {k: v for k, v in state_dict.items() if hasattr(v, "detach")}
    dropped = [k for k in params if k in NON_PARAMETER_BUFFERS]
    for key in dropped:
        params.pop(key)
    return params, dropped


def _remap_proj_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """把官方的 text.proj.0.* / text.proj.2.* 映射到 Sequential 的索引命名。"""
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("text.proj."):
            parts = key.split(".")
            idx = {"0": "0", "2": "2"}.get(parts[2], parts[2])
            out[f"text.proj.seq.{idx}.{'.'.join(parts[3:])}"] = value
        else:
            out[key] = value
    return out


def tensor_digest(state_dict: dict[str, torch.Tensor]) -> tuple[str, int, int]:
    h = hashlib.sha256()
    n_params = 0
    for key in sorted(state_dict):
        arr = state_dict[key].detach().cpu().numpy()
        h.update(key.encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(",".join(map(str, arr.shape)).encode("utf-8"))
        h.update(arr.tobytes())
        n_params += int(arr.size)
    return h.hexdigest(), len(state_dict), n_params


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class LoadedModel:
    model: nn.Module
    state_dict: dict[str, torch.Tensor]
    fingerprint: dict[str, Any]
    image_transform: Any = None
    tokenizer: Any = None
    device: str = "cuda"


def load_biomedclip(
    weights_dir: str | Path,
    device: str = "cuda",
    image_transform: Any = None,
    tokenizer: Any = None,
) -> LoadedModel:
    """Load the real BiomedCLIP weights into a freshly built module."""
    weights_dir = Path(weights_dir)
    config_path = weights_dir / "open_clip_config.json"
    weight_path = weights_dir / "open_clip_pytorch_model.bin"
    if not weight_path.exists():
        raise FileNotFoundError(f"missing weights: {weight_path}")

    raw_cfg = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    model = BiomedCLIP()

    state_dict, dropped = _checkpoint_state_dict(weight_path)
    state_dict = _remap_proj_keys(state_dict)
    load_missing_unexpected(model, state_dict)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dev = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    model = model.to(dev)

    file_hash = file_sha256(weight_path)
    tensor_hash, n_tensors, n_params = tensor_digest(state_dict)
    fingerprint = {
        "official_hf_id": OFFICIAL_HF_ID,
        "arch_version": ARCH_VERSION,
        "weights_path": str(weight_path),
        "weights_bytes": weight_path.stat().st_size,
        "file_sha256": file_hash,
        "tensor_sha256": tensor_hash,
        "n_tensors": n_tensors,
        "n_parameters": n_params,
        "load_mode": "strict zero-missing zero-unexpected",
        "dropped_non_parameter_buffers": dropped,
        "dropped_reason": "position_ids 是 HF BertModel 的常量 buffer（不是可学习参数），本实现用 torch.arange 等价生成",
        "context_length": model.text_cfg.context_length,
        "image_size": model.vision_cfg.image_size,
        "embed_dim": EMBED_DIM,
        "logit_scale": float(model.logit_scale.detach().cpu()),
        "logit_scale_exp": float(model.logit_scale.detach().exp().cpu()),
        "open_clip_config": raw_cfg,
        "device": dev,
        "torch_version": torch.__version__,
    }
    return LoadedModel(
        model=model,
        state_dict=state_dict,
        fingerprint=fingerprint,
        image_transform=image_transform,
        tokenizer=tokenizer,
        device=dev,
    )


# ---------------------------------------------------------------------------
# 推理入口
# ---------------------------------------------------------------------------
@torch.no_grad()
def embed_images(loaded: LoadedModel, image_paths: list[str | Path], batch_size: int = 16) -> np.ndarray:
    from PIL import Image

    model = loaded.model
    chunks = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        tensors = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            arr = loaded.image_transform(img)  # ImageTransform 返回 (3,H,W) float32 的 numpy
            tensors.append(torch.from_numpy(np.ascontiguousarray(arr)))
        x = torch.stack(tensors).to(loaded.device)
        chunks.append(model.encode_image(x).float().cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, EMBED_DIM), dtype=np.float32)


@torch.no_grad()
def embed_texts(loaded: LoadedModel, texts: list[str], batch_size: int = 32) -> np.ndarray:
    model = loaded.model
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        input_ids, attention_mask = loaded.tokenizer.batch_encode(batch, model.text_cfg.context_length)
        ids = torch.from_numpy(input_ids.astype(np.int64)).to(loaded.device)
        mask = torch.from_numpy(attention_mask.astype(np.int64)).to(loaded.device)
        chunks.append(model.encode_text(ids, mask).float().cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, EMBED_DIM), dtype=np.float32)
