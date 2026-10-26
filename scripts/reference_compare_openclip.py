#!/usr/bin/env python
"""可移植的「官方 open_clip 参考实现 ↔ 本项目自包含实现」逐向量对照脚本。

用途
----
本项目开发机（Windows / torch 1.13.1+cu116）无法运行官方 open_clip：
  - medalign venv：`import open_clip` 直接段错误（exit 139）
  - 新建 refclip venv（torch 2.x CPU）：`c10.dll` DLL 初始化失败（WinError 1114）
因此"实时逐向量对照"在本机记为 BLOCKED。本脚本供**在满足依赖的机器上**一键完成该对照，
把结论从"未验证"升级为"已验证/已证伪"。

依赖（需在目标机器安装）
------------------------
    pip install torch open_clip_torch timm transformers pillow numpy

用法
----
    python scripts/reference_compare_openclip.py \
        --weights-dir data/external/models/biomedclip \
        --manifest   data/metadata/clean_manifest.jsonl \
        --images-dir data/raw/images \
        --n 32 --out reports/reference_compare_openclip.json

比对内容
--------
1. 图像塔：同批图像 → 官方 vs 本项目 embedding，报告 max_abs_diff / 余弦相似度。
2. 文本塔：同批图注 → 官方 tokenizer vs 本项目 tokenizer 的 input_ids 是否**逐位相同**，
   以及两侧 text embedding 的 max_abs_diff / 余弦。
3. 检索层面：两侧相似度矩阵的 Top-1 命中集合一致率（rank agreement）。
4. 若任一项不一致，脚本退出码 1 并在 JSON 中标注具体差异，便于定位真实计算差异。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_manifest(p: Path, n: int):
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", default="data/external/models/biomedclip")
    ap.add_argument("--manifest", default="data/metadata/clean_manifest.jsonl")
    ap.add_argument("--images-dir", default="data/raw/images")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--out", default="reports/reference_compare_openclip.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    try:
        import torch
        import open_clip
        from PIL import Image
    except Exception as e:  # pragma: no cover
        print(f"[BLOCKED] official deps unavailable: {e}")
        Path(args.out).write_text(json.dumps({
            "status": "BLOCKED",
            "reason": f"official open_clip stack unavailable on this machine: {e}",
            "how_to_run": "run this script on a machine with: pip install torch open_clip_torch timm transformers pillow numpy",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return 2

    from medalign.models.biomedclip import load_biomedclip as load_mine
    from medalign.models.preprocessing import BertWordPieceTokenizer, ImageTransform

    wdir = Path(args.weights_dir)
    rows = load_manifest(Path(args.manifest), args.n)
    if not rows:
        print("[error] empty manifest")
        return 1

    dev = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[info] device={dev} n={len(rows)}", flush=True)

    # ---------- 官方 ----------
    model_off, _, preprocess_off = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    tok_off = open_clip.get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    model_off = model_off.to(dev).eval()

    # ---------- 本项目 ----------
    mine = load_mine(str(wdir), device=dev)
    my_tok = BertWordPieceTokenizer(wdir / "vocab.txt")
    my_tf = ImageTransform(224)

    captions = [r.get("caption") or "" for r in rows]
    img_paths = [Path(args.images_dir) / f"{r['sample_id']}.jpg" for r in rows]

    # 文本：token id 逐位比对
    ids_off = tok_off(captions)
    ids_mine, _ = my_tok.batch_encode(captions, 256)
    ids_off_np = ids_off.detach().cpu().numpy()
    L = min(ids_off_np.shape[1], ids_mine.shape[1])
    exact_rows = int((ids_off_np[:, :L] == ids_mine[:, :L]).all(axis=1).sum())
    token_id_identical = bool(exact_rows == len(rows))

    # 图像张量
    imgs = [Image.open(p).convert("RGB") for p in img_paths]
    x_off = torch.stack([preprocess_off(im) for im in imgs]).to(dev)
    x_mine = torch.from_numpy(np.stack([my_tf(im) for im in imgs])).to(dev)

    with torch.no_grad():
        e_img_off = model_off.encode_image(x_off).float().cpu().numpy()
        e_txt_off = model_off.encode_text(ids_off.to(dev)).float().cpu().numpy()
        e_img_mine = mine.model.encode_image(x_mine).float().cpu().numpy()
        ids_mine_t = torch.from_numpy(ids_mine).to(dev)
        mask_mine_t = torch.from_numpy((ids_mine != 0).astype(np.int64)).to(dev)
        e_txt_mine = mine.model.encode_text(ids_mine_t, mask_mine_t).float().cpu().numpy()

    def cmp(a, b):
        d = np.abs(a - b)
        a_n = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_n = b / np.linalg.norm(b, axis=1, keepdims=True)
        cos = float((a_n * b_n).sum(axis=1).mean())
        return {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()), "mean_cosine": cos}

    img_cmp = cmp(e_img_off, e_img_mine)
    txt_cmp = cmp(e_txt_off, e_txt_mine)

    # 检索层面：Top-1 一致率
    def top1(img_e, txt_e):
        ii = img_e / np.linalg.norm(img_e, axis=1, keepdims=True)
        tt = txt_e / np.linalg.norm(txt_e, axis=1, keepdims=True)
        return (ii @ tt.T).argmax(axis=1)

    t1_off, t1_mine = top1(e_img_off, e_txt_off), top1(e_img_mine, e_txt_mine)
    rank_agreement = float((t1_off == t1_mine).mean())

    tol = 1e-3
    ok = (img_cmp["max_abs_diff"] < tol and txt_cmp["max_abs_diff"] < tol
          and token_id_identical and rank_agreement == 1.0)

    out = {
        "status": "PASS" if ok else "DIFFERENCE_DETECTED",
        "device": dev,
        "n_samples": len(rows),
        "tolerance_max_abs_diff": tol,
        "tokenizer": {
            "token_id_rows_identical": int(exact_rows),
            "n_rows": len(rows),
            "identical": token_id_identical,
        },
        "image_tower": img_cmp,
        "text_tower": txt_cmp,
        "retrieval_top1_agreement": rank_agreement,
        "note": ("若 status=PASS 则自包含前向与官方实现在数值上等价（容差内）；"
                 "若 DIFFERENCE_DETECTED 则按上述分项定位真实计算差异。"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
