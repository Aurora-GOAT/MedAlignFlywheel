#!/usr/bin/env python
"""P0：从 ROCOv2 官方 HuggingFace 仓库摄取真实医学图文配对子集。

用法：
    python scripts/run_p0_ingest.py --out-root . \
        --n-blocks-train 20 --take-train 30 \
        --n-blocks-val 8 --take-val 25 \
        --n-blocks-test 8 --take-test 25 --seed 20260701
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medalign.pipeline.roco_ingest import IngestConfig, run_ingest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=".")
    ap.add_argument("--seed", type=int, default=20260701)
    ap.add_argument("--n-blocks-train", type=int, default=20)
    ap.add_argument("--take-train", type=int, default=30)
    ap.add_argument("--n-blocks-val", type=int, default=8)
    ap.add_argument("--take-val", type=int, default=25)
    ap.add_argument("--n-blocks-test", type=int, default=8)
    ap.add_argument("--take-test", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = IngestConfig(
        out_root=args.out_root,
        seed=args.seed,
        n_blocks_train=args.n_blocks_train,
        take_train=args.take_train,
        n_blocks_val=args.n_blocks_val,
        take_val=args.take_val,
        n_blocks_test=args.n_blocks_test,
        take_test=args.take_test,
        workers=args.workers,
    )
    result = run_ingest(cfg)
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
