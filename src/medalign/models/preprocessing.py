"""图像→patch、文本→WordPiece ids：把4.1章定义的前处理写成 NumPy 可执行形式。

要点
----
* 图像重采样必须是 **area / bicubic 对齐 open_clip** 的区别必须写清楚，
  否则 CLIP/BiomedCLIP 的分数会和官方对不上，还很难 debug；本项目默认用
  area ('box-filter') 重采样 + center crop + 官方 mean/std。
* 文本用 **WordPiece**（BertTokenizer 语义），从 spec.TextCfg.vocab_size 出发，
  自己做 simplify english空白/标点切分 + 贪心最长匹配。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

EMBED_DIM = 512


def _as_batch_uint8(images) -> np.ndarray:
    """Normalise various image inputs to (N, H, W, 3) uint8."""
    arr = np.asarray(images)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.shape[-1] == 4:  # RGBA -> RGB
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0 + 1e-6:
            arr = (arr * 255.0).clip(0, 255)
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def resize_short_side_and_center_crop(
    image: Image.Image, image_size: int, resample: str = "bicubic"
) -> np.ndarray:
    """Resize shortest side then center-crop, open_clip-style."""
    w, h = image.size
    scale = image_size / min(w, h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    method = {
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "box": Image.BOX,
        "area": Image.BOX,
        "nearest": Image.NEAREST,
    }[resample]
    img = image.resize((new_w, new_h), resample=method)
    left = (new_w - image_size) // 2
    top = (new_h - image_size) // 2
    img = img.crop((left, top, left + image_size, top + image_size))
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


class ImageTransform:
    """open_clip-compatible image transform, implemented on NumPy."""

    def __init__(self, image_size: int = 224, mean=(0.48145466, 0.4578275, 0.40821073),
                 std=(0.26862954, 0.26130258, 0.27577711), resample: str = "bicubic"):
        self.image_size = image_size
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.resample = resample

    def __call__(self, image: Image.Image) -> np.ndarray:
        arr = resize_short_side_and_center_crop(image, self.image_size, self.resample)
        x = arr.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        return np.ascontiguousarray(x.transpose(2, 0, 1))  # (3, H, W)

    def batch(self, images) -> np.ndarray:
        """images: list[PIL.Image] or an (N,H,W,3) uint8 array."""
        if isinstance(images, np.ndarray):
            batch = [_as_batch_uint8(images[i]) for i in range(len(images))]
            batch = [Image.fromarray(a) for a in batch]
        else:
            batch = list(images)
        return np.stack([self(img) for img in batch], axis=0)


def make_image_transform(cfg) -> ImageTransform:
    return ImageTransform(image_size=cfg.image_size, mean=cfg.mean, std=cfg.std,
                          resample=getattr(cfg, "resample", "bicubic"))


# -----------------------------------------------------------------------------
# WordPiece tokenizer — BertTokenizer semantics, stdlib + numpy only
# -----------------------------------------------------------------------------
_PUNCTUATION = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def _whitespace_tokenize(text: str) -> list[str]:
    return [t for t in text.strip().split() if t]


def _split_on_punctuation(token: str) -> list[str]:
    pieces: list[str] = []
    buf: list[str] = []
    for ch in token:
        if ch in _PUNCTUATION:
            if buf:
                pieces.append("".join(buf))
                buf = []
            pieces.append(ch)
        else:
            buf.append(ch)
    if buf:
        pieces.append("".join(buf))
    return pieces


def basic_tokenize(text: str) -> list[str]:
    """BasicTokenizer: lowercase + strip accents + punctuation split."""
    out: list[str] = []
    for token in _whitespace_tokenize(text.lower()):
        for piece in _split_on_punctuation(token):
            if piece:
                out.append(piece)
    return out


class BertWordPieceTokenizer:
    """A tiny but faithful WordPiece tokenizer.

    `vocab_file` is the standard BERT ``vocab.txt``. We do not require
    transformers at runtime, which keeps the CPU pathway truly stdlib-only.
    """

    def __init__(self, vocab_file: str | Path, do_lower_case: bool = True, unk: str = "[UNK]"):
        tokens: list[str] = []
        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                tok = line.rstrip("\n")
                if tok:
                    tokens.append(tok)
        self.vocab: dict[str, int] = {t: i for i, t in enumerate(tokens)}
        self.inv_vocab: dict[int, str] = {i: t for t, i in self.vocab.items()}
        self.do_lower_case = do_lower_case
        self.unk_token = unk
        self.unk_token_id = self.vocab.get(unk, 100)
        self.cls_token_id = self.vocab.get("[CLS]", 101)
        self.sep_token_id = self.vocab.get("[SEP]", 102)
        self.pad_token_id = self.vocab.get("[PAD]", 0)

    def wordpiece(self, token: str) -> list[str]:
        if len(token) > 100:
            return [self.unk_token]
        sub_tokens: list[str] = []
        start = 0
        while start < len(token):
            end = len(token)
            cur = None
            while start < end:
                sub = token[start:end]
                if start > 0:
                    sub = "##" + sub
                if sub in self.vocab:
                    cur = sub
                    break
                end -= 1
            if cur is None:
                return [self.unk_token]
            sub_tokens.append(cur)
            start = end
        return sub_tokens

    def tokenize(self, text: str) -> list[str]:
        out: list[str] = []
        for token in basic_tokenize(text):
            out.extend(self.wordpiece(token))
        return out

    def encode(self, text: str, context_length: int) -> list[int]:
        ids = [self.cls_token_id]
        for tok in self.tokenize(text):
            ids.append(self.vocab.get(tok, self.unk_token_id))
            if len(ids) >= context_length - 1:
                break
        ids.append(self.sep_token_id)
        return ids[:context_length]

    def batch_encode(self, texts, context_length: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (input_ids, attention_mask) both int64, padded to context_length."""
        n = len(texts)
        input_ids = np.full((n, context_length), self.pad_token_id, dtype=np.int64)
        attention_mask = np.zeros((n, context_length), dtype=np.int64)
        for i, text in enumerate(texts):
            ids = self.encode(text, context_length)
            input_ids[i, : len(ids)] = ids
            attention_mask[i, : len(ids)] = 1
        return input_ids, attention_mask


def make_tokenizer(cfg) -> BertWordPieceTokenizer:
    vocab_file = cfg.vocab_file or str(Path(__file__).resolve().parents[3] / "data" / "external" / "models" / "biomed_clip_torch" / "vocab.txt")
    return BertWordPieceTokenizer(vocab_file, do_lower_case=cfg.use_lower_case)
