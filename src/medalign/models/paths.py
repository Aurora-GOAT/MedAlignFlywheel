"""Model path resolution ("genie cache").

Keeps everything in one place so that every module agrees on where a model
snapshot lives. Paths are handled with pathlib only, which keeps Windows paths
containing non-ASCII characters stable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from ..dataio import PROJECT_ROOT

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_STATE_PATH = CACHE_DIR / "file-path-cache.json"


@dataclass(frozen=True)
class ResolvedPath:
    """Result of one lookup.

    root: first candidate that exists on disk (None when nothing was found)
    hits: per-candidate existence flags, useful for debugging
    """

    key: str
    root: Path | None
    hits: dict[str, bool]

    @property
    def ok(self) -> bool:
        return self.root is not None


def _load_state() -> dict:
    if CACHE_STATE_PATH.exists():
        try:
            return json.loads(CACHE_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a broken cache is just an empty cache
            return {}
    return {}


def _save_state(data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_STATE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def find_one_file(candidates: list[str], key: str, use_cache: bool = True) -> ResolvedPath:
    """Pick the first candidate path that exists.

    The decision is memoized per key so repeated script runs do not rescan disk.
    """
    hits: dict[str, bool] = {}
    chosen: Path | None = None
    for cand in candidates:
        p = Path(os.path.expandvars(str(cand))).expanduser()
        exists = p.exists()
        hits[str(p)] = exists
        if exists and chosen is None:
            chosen = p
    if use_cache:
        state = _load_state()
        state[key] = {"hits": hits, "chosen": str(chosen) if chosen else None}
        _save_state(state)
    return ResolvedPath(key=key, root=chosen, hits=hits)
