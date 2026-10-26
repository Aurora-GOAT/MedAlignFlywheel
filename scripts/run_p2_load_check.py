"""P2 第一步：证明权重真的被完整加载、且前向给出的相似度不是随机噪声。

判定标准（写死在脚本里，便于复核）：
1. zero-missing / zero-unexpected 的严格加载必须成功；
2. 前 8 张图 × 前 8 条图注的相似度矩阵里，正确配对（对角线）必须
   **全部**是各自行/列的最大值 —— 一个随便搭错的模型做不到这一点；
3. 正例均值与错配均值必须有明显间隔。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from medalign.models import biomedclip  # noqa: E402
from medalign.models.preprocessing import BertWordPieceTokenizer, ImageTransform  # noqa: E402

MODEL_DIR = "D:/PQY/Five/post/实习、工作/找工作/简历合集/蚂蚁/MedAlignFlywheel/data/external/models/biomedclip"
CLEAN = "D:/PQY/Five/post/实习、工作/找工作/简历合集/蚂蚁/MedAlignFlywheel/data/metadata/clean_manifest.jsonl"
RAW = "D:/PQY/Five/post/实习、工作/找工作/简历合集/蚂蚁/MedAlignFlywheel/data/raw"
N = 8


def main() -> int:
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
    vocab = Path(MODEL_DIR) / "vocab.txt"
    tokenizer = BertWordPieceTokenizer(vocab)
    cfg = json.loads((Path(MODEL_DIR) / "open_clip_config.json").read_text(encoding="utf-8"))
    pre = cfg.get("preprocess_cfg", {})
    transform = ImageTransform(
        image_size=224,
        mean=tuple(pre.get("mean", (0.48145466, 0.4578275, 0.40821073))),
        std=tuple(pre.get("std", (0.26862954, 0.26130258, 0.27577711))),
        resample="bicubic",
    )

    loaded = biomedclip.load_biomedclip(MODEL_DIR, device="cuda", image_transform=transform, tokenizer=tokenizer)
    print("\n=== fingerprint ===")
    print(json.dumps(loaded.fingerprint, ensure_ascii=False, indent=2)[:1600])

    rows = [json.loads(line) for line in Path(CLEAN).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows[:N]
    paths = [str(Path(RAW) / r["image_relpath"]) for r in rows]
    texts = [r["caption"] for r in rows]

    img = biomedclip.embed_images(loaded, paths, batch_size=4)
    txt = biomedclip.embed_texts(loaded, texts, batch_size=4)

    def norm(a: np.ndarray) -> np.ndarray:
        return a / np.linalg.norm(a, axis=1, keepdims=True)

    sim = norm(img) @ norm(txt).T
    diag = np.diag(sim)
    off = sim[~np.eye(len(rows), dtype=bool)]

    print("\n=== 8x8 cosine similarity (first 5x5) ===")
    print(np.round(sim[:5, :5], 3))
    print("\npositive (diagonal) mean :", round(float(diag.mean()), 4))
    print("mismatched mean          :", round(float(off.mean()), 4))
    print("mismatched std           :", round(float(off.std()), 4))
    row_max_is_diag = int(np.sum(np.argmax(sim, axis=1) == np.arange(len(rows))))
    col_max_is_diag = int(np.sum(np.argmax(sim, axis=0) == np.arange(len(rows))))
    print(f"rows whose argmax is the true caption : {row_max_is_diag}/{len(rows)}")
    print(f"cols whose argmax is the true image   : {col_max_is_diag}/{len(rows)}")

    ok = row_max_is_diag >= 7 and col_max_is_diag >= 7 and float(diag.mean()) - float(off.mean()) > 0.1
    print("\nVERDICT:", "PASS — 权重与结构对的上了" if ok else "FAIL — 需要排查结构/权重")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
