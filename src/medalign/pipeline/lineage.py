"""数据血缘记录。

每个被处理的样本都要能回答四个问题：从哪来、被谁处理过、为什么留下/丢掉、当时用的什么参数。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

PIPELINE_VERSION = "medalign-quality-pipeline-v1.1"


@dataclass
class LineageEntry:
    sample_id: str
    source_dataset: str
    source_split: str
    source_row_idx: Any
    upstream_image_url: str
    image_sha256: str
    pipeline_version: str = PIPELINE_VERSION
    operators: list[dict] = field(default_factory=list)
    decision: str = "keep"
    reject_codes: list[str] = field(default_factory=list)
    warn_codes: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)
