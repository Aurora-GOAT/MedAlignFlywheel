#!/usr/bin/env python
"""P2：用真实 BiomedCLIP 权重提取全量图文向量（GPU 推理，结果落盘缓存）。

输出：
- data/cache/emb_<split>.npz : image_emb (N,512) / text_emb (N,512)
- data/cache/emb_<split>.ids.json : 与行一一对应的 sample_id 列表
- reports/embedding_run.json : 运行环境、指纹、耗时

向量缓存是复用的：P3 难例挖掘与 P4 消融都从这里读，不重复推理。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from medalign.models import biomedclip  # noqa: E402
from medalign.models.preprocessing import BertWordPieceTokenizer, ImageTransform  # noqa: E402
from medalign.models.torch_utils import collect_env_fingerprint, pick_device, set_all_seeds  # noqa: E402


def load_manifest(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/metadata/clean_manifest.jsonl")
    ap.add_argument("--raw-root", default="data/raw")
    ap.add_argument("--weights-dir", default="data/external/models/biomedclip")
    ap.add_argument("--out-dir", default="data/cache")
    ap.add_argument("--batch-image", type=int, default=32)
    ap.add_argument("--batch-text", type=int, default=64)
    ap.add_argument("--splits", nargs="*", default=["train", "validation", "test"])
    ap.add_argument("--seed", type=int, default=20260701)
    args = ap.parse_args()

    seeds = set_all_seeds(args.seed)
    device = pick_device(True)
    print(f"[device] {device}  [seed] {seeds}", flush=True)

    import json as _json

    cfg = _json.loads((Path(args.weights_dir) / "open_clip_config.json").read_text(encoding="utf-8"))
    pre = cfg.get("preprocess_cfg", {})
    transform = ImageTransform(
        image_size=224,
        mean=tuple(pre.get("mean", (0.48145466, 0.4578275, 0.40821073))),
        std=tuple(pre.get("std", (0.26862954, 0.26130258, 0.27577711))),
    )
    tokenizer = BertWordPieceTokenizer(Path(args.weights_dir) / "vocab.txt")
    bundle = biomedclip.load_biomedclip(
        args.weights_dir, device=device, image_transform=transform, tokenizer=tokenizer
    )
    print(
        "[loaded] file_sha256=",
        bundle.fingerprint["file_sha256"][:16],
        " params=",
        bundle.fingerprint["n_parameters"],
        " tensors=",
        bundle.fingerprint["n_tensors"],
        flush=True,
    )

    rows = load_manifest(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_info = {
        "seed": seeds,
        "env": collect_env_fingerprint(device),
        "weight_fingerprint": bundle.fingerprint,
        "manifest": args.manifest,
        "splits": {},
        "batch_size": {"image": args.batch_image, "text": args.batch_text},
        "note": "embeddings are L2-unormalized raw projections; retrieval code normalizes",
    }

    for split in args.splits:
        sub = [r for r in rows if r.get("split") == split and r.get("image_relpath")]
        if not sub:
            print(f"[{split}] empty, skip", flush=True)
            continue
        paths = [Path(args.raw_root) / r["image_relpath"] for r in sub]
        paths = [p for p in paths if p.exists()]
        captions = [r["caption"] for r in sub]

        t0 = time.time()
        img_emb = biomedclip.embed_images(bundle, paths, batch_size=args.batch_image)
        t_img = time.time() - t0

        t0 = time.time()
        txt_emb = biomedclip.embed_texts(bundle, captions, batch_size=args.batch_text)
        t_txt = time.time() - t0

        np.savez(out_dir / f"emb_{split}.npz", image_emb=img_emb, text_emb=txt_emb)
        (out_dir / f"emb_{split}.ids.json").write_text(
            json.dumps([r["sample_id"] for r in sub], ensure_ascii=False), encoding="utf-8"
        )
        run_info["splits"][split] = {
            "n_samples": len(sub),
            "image_emb_shape": list(img_emb.shape),
            "text_emb_shape": list(txt_emb.shape),
            "image_sec": round(t_img, 2),
            "text_sec": round(t_txt, 2),
            "images_per_sec": round(len(paths) / max(t_img, 1e-6), 2),
        }
        print(
            f"[{split}] n={len(sub)} image={list(img_emb.shape)} {t_img:.1f}s  "
            f"text={list(txt_emb.shape)} {t_txt:.1f}s",
            flush=True,
        )

    Path("reports").mkdir(exist_ok=True)
    Path("reports/embedding_run.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[saved] reports/embedding_run.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
