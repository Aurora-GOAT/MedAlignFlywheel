#!/usr/bin/env python
"""P2 冒烟测试：加载真实 BiomedCLIP 权重，跑一次 GPU 图文推理。

这一步的目的不是出指标，而是回答四个"能不能信"的问题：
1. 权重能不能被完整加载（strict + zero missing/unexpected）；
2. 出来的向量是不是非零、有区分度（不是随机初始化）；
3. zero-shot 语义是否成立（.isArrayEqual 官方 example 的模态分类）；
4. tokenizer 的 WordPiece 是否工作。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from medalign.models import biomedclip  # noqa: E402
from medalign.models.torch_utils import collect_env_fingerprint, pick_device  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", default="data/external/models/biomedclip")
    ap.add_argument("--image-dir", default="data/raw/images")
    ap.add_argument("--meta", default="data/metadata/raw_manifest.jsonl")
    ap.add_argument("--n-images", type=int, default=4)
    ap.add_argument("--out", default="data/metadata/p2_smoke.json")
    args = ap.parse_args()

    device = pick_device(True)
    print(f"[device] {device}")
    bundle = biomedclip.load_biomedclip(args.weights_dir, device=device)
    print("[fingerprint]", json.dumps(bundle.fingerprint, ensure_ascii=False, indent=2))

    # --- tokenizer 自检 ---
    toks = bundle.tokenizer.tokenize("Chest X-ray showing bilateral pleural effusion.")
    print(f"[tokenizer] n={len(toks)} tokens -> {toks[:12]}")

    # --- 真实图像 embedding ---
    import itertools

    image_dir = Path(args.image_dir)
    paths = sorted(itertools.islice(image_dir.glob("*.jpg"), args.n_images))
    paths = list(paths) + sorted(itertools.islice(image_dir.glob("*.png"), max(0, args.n_images - len(paths))))
    if not paths:
        raise SystemExit("no local images found; run P0 ingest first")

    img_emb = biomedclip.embed_images(bundle, paths, batch_size=2)
    norms = np.linalg.norm(img_emb, axis=1)
    print(f"[image emb] shape={img_emb.shape} norm mean={norms.mean():.4f} std={norms.std():.4f}")
    eye = (img_emb @ img_emb.T) / np.outer(norms, norms)
    print(f"[image self-sim] diag≈1: {np.round(np.diag(eye), 3)}, offdiag max={np.max(eye - np.eye(len(paths))):.3f}")

    # --- 文本 embedding（用真实图注） ---
    captions = []
    for line in Path(args.meta).read_text(encoding="utf-8").splitlines()[: args.n_images]:
        if line.strip():
            import json as _json

            captions.append(_json.loads(line)["caption"])
    txt_emb = biomedclip.embed_texts(bundle, captions + ["a completely unrelated banana recipe"], batch_size=8)
    print(f"[text emb] shape={txt_emb.shape}")

    # --- zero-shot 语义冒烟：同一张图送五个模态候选句 ---
    probe = [
        "chest x-ray radiograph of the thorax",
        "axial computed tomography of the abdomen",
        "T2 weighted magnetic resonance image of the brain",
        "histopathology micrograph with hematoxylin and eosin",
        "a banana",
    ]
    probe_emb = biomedclip.embed_texts(bundle, probe, batch_size=8)
    img0 = img_emb[0:1] / np.linalg.norm(img_emb[0:1])
    probe_norm = probe_emb / np.linalg.norm(probe_emb, axis=1, keepdims=True)
    scores = (img0 @ probe_norm.T).ravel()
    print(f"[zero-shot probe] image={paths[0].name}")
    for p, s in sorted(zip(probe, scores.tolist()), key=lambda kv: -kv[1]):
        print(f"   {s:+.4f}  {p}")

    env = collect_env_fingerprint(device)
    out = {
        "device": device,
        "env": env,
        "fingerprint": bundle.fingerprint,
        "tokenizer_sample": toks[:16],
        "image_files": [p.name for p in paths],
        "image_emb_shape": list(img_emb.shape),
        "image_norm_mean": float(norms.mean()),
        "zero_shot_probe": sorted(
            [{"text": p, "score": float(s)} for p, s in zip(probe, scores.tolist())],
            key=lambda d: -d["score"],
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
