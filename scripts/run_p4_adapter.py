#!/usr/bin/env python
"""P4：冻结 BiomedCLIP 编码器，只训小对齐头，按 A/B/C/D 同预算对照（冻结 test 评估）。

纪律：
- 对齐头只在各策略选出的 **train 子集** 上训练；
- 评估只在 **冻结 test** 上做，test 不参与任何训练；
- 输出是"数据选择策略"的对照结果，不是"模型能力证明"；
- 若差异在噪声内，如实写明，不夸大。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from medalign.flywheel.adapter import (  # noqa: E402
    AdapterConfig,
    baseline_metrics,
    evaluate_adapter,
    train_adapter,
)


def load_emb(split: str, cache_dir: Path):
    d = np.load(cache_dir / f"emb_{split}.npz", allow_pickle=True)
    ids = json.loads((cache_dir / f"emb_{split}.ids.json").read_text(encoding="utf-8"))
    return d["image_emb"].astype(np.float32), d["text_emb"].astype(np.float32), ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--meta-dir", default="data/metadata")
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260701)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--strategies", nargs="*",
                    default=["A_random", "B_quality", "C_quality_hard", "D_quality_longtail"])
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    meta_dir = Path(args.meta_dir)
    train_img, train_txt, train_ids = load_emb("train", cache_dir)
    test_img, test_txt, test_ids = load_emb("test", cache_dir)

    train_by_id = {sid: i for i, sid in enumerate(train_ids)}

    cfg = AdapterConfig(seed=args.seed, device=args.device, steps=400, batch_size=64)

    out: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "adapter_config": {
            "hidden": cfg.hidden, "steps": cfg.steps, "lr": cfg.lr,
            "batch_size": cfg.batch_size, "temperature_init": cfg.temperature_init,
        },
        "disclaimer": (
            "冻结 BiomedCLIP 编码器已在本任务上接近天花板（test I2T R@1≈0.81）。"
            "仅训 200 样本小对齐头对数据选择策略的区分力有限；本实验如实报告，不夸大差异。"
        ),
        "baseline": {},
        "strategies": {},
    }

    # 基线（不训练）
    base = baseline_metrics(test_img, test_txt)
    out["baseline"] = {
        "i2t": base["i2t"], "t2i": base["t2i"],
        "note": "直接使用原始 BiomedCLIP 向量的冻结基线",
    }
    print(f"[baseline] I2T R@1={base['i2t']['recall@1']:.4f} T2I R@1={base['t2i']['recall@1']:.4f}", flush=True)

    for strat in args.strategies:
        sel_path = meta_dir / f"selected_{strat}.json"
        if not sel_path.exists():
            print(f"[{strat}] 未找到选择结果，跳过", flush=True)
            continue
        sel = json.loads(sel_path.read_text(encoding="utf-8"))
        sel_ids = sel["sample_ids"]
        idx = [train_by_id[s] for s in sel_ids if s in train_by_id]
        if not idx:
            print(f"[{strat}] 无可用训练向量，跳过", flush=True)
            continue
        sub_img = train_img[idx]
        sub_txt = train_txt[idx]

        head, temperature, log = train_adapter(sub_img, sub_txt, cfg)
        res = evaluate_adapter(head, temperature, test_img, test_txt)
        out["strategies"][strat] = {
            "n_train_used": log["n_train"],
            "adapter_train_log": log,
            "temperature": round(temperature, 4),
            "i2t": res["i2t"],
            "t2i": res["t2i"],
        }
        print(
            f"[{strat}] n={log['n_train']} temp={temperature:.2f} "
            f"I2T R@1={res['i2t']['recall@1']:.4f} T2I R@1={res['t2i']['recall@1']:.4f}",
            flush=True,
        )

    Path("reports").mkdir(exist_ok=True)
    Path("reports/p4_adapter.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[saved] reports/p4_adapter.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
