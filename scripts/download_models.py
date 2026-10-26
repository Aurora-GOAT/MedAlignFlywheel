"""HF model snapshot downloader.

Deliberately minimal: the download must go where the project can see it and the
HTTP layer must not be hidden inside a dozen wrappers.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out", required=True, help="local snapshot directory")
    ap.add_argument(
        "--patterns",
        nargs="*",
        default=["*.json", "*.txt", "*.bin"],
        help="only these file globs are pulled from the repo",
    )
    ap.add_argument("--max-workers", type=int, default=4)
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is unavailable here.\n"
            "  * baseline env:  /usr/bin/python3 (has huggingface_hub 0.35.0)\n"
            "  * pixi env:      pixi run python  (may need install; torch dl in flight)",
            file=sys.stderr,
        )
        return 2

    t0 = time.time()
    path = snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=args.out,
        max_workers=args.max_workers,
        allow_patterns=args.patterns,
    )
    dt = time.time() - t0
    print(f"[OK] snapshot -> {path}")
    print(f"[OK] elapsed  -> {dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
