#!/usr/bin/env python
"""P1：跑完整质量管线，产出 clean/rejected/血缘/重复簇/质量报告。"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medalign.pipeline.runner import QualityPipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/metadata/raw_manifest.jsonl")
    ap.add_argument("--raw-root", default="data/raw")
    ap.add_argument("--meta-dir", default="data/metadata")
    ap.add_argument("--config", default="configs/quality_ops.json")
    args = ap.parse_args()

    cfg = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        import json

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    pipe = QualityPipeline(
        raw_root=args.raw_root,
        meta_dir=args.meta_dir,
        image_thresholds=cfg.get("image", {}),
        text_thresholds=cfg.get("text", {}),
        dedup_thresholds=cfg.get("dedup", {}),
    )
    report = pipe.run(args.manifest)
    print("\n=== P1 QUALITY REPORT SUMMARY ===")
    for key in (
        "n_total",
        "n_keep",
        "n_reject",
        "keep_rate",
        "by_split_total",
        "by_split_keep",
        "reject_code_counts",
        "warn_code_counts",
        "duplicate_clusters",
        "modality_proxy_total",
        "cui_stats",
    ):
        print(f"{key}: {report.get(key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
