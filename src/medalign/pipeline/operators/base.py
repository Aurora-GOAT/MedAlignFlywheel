"""算子基类与结果结构。

设计约束（本项目红线）：
- 每条Finding都必须能说出**判定码 + 触发阈值 + 命中的证据**，便于人工复核；
- 任何算子都不能声称"医学上正确"，只能声称"满足/不满足某个字面规则"；
- reject 的语义是"从本轮实验候选中剔除"，不等于"这条数据本身是坏的"。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

OK = "ok"
WARN = "warn"
REJECT = "reject"


@dataclass
class Finding:
    op: str
    version: str
    severity: str
    code: str
    message: str
    evidence: dict = field(default_factory=dict)
    threshold: Any = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OpResult:
    sample_id: str
    findings: list[Finding] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def rejected(self) -> bool:
        return any(f.severity == REJECT for f in self.findings)

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == WARN]

    def reject_codes(self) -> list[str]:
        return [f.code for f in self.findings if f.severity == REJECT]

    def warn_codes(self) -> list[str]:
        return [f.code for f in self.findings if f.severity == WARN]


class Operator:
    """所有算子继承此类；name / version 必须写死，不得动态生成（血缘可复现）。"""

    name: str = "base"
    version: str = "0.0"
    # 该算子只产出代理指标（不是医学判定）
    proxy_only: bool = True

    def run(self, record: dict, ctx: dict) -> OpResult:
        raise NotImplementedError


def merge_results(sample_id: str, results: Iterable[OpResult]) -> OpResult:
    merged = OpResult(sample_id=sample_id)
    for r in results:
        merged.findings.extend(r.findings)
        merged.metrics.update(r.metrics)
    return merged


def count_findings(results: Iterable[OpResult]) -> dict[str, int]:
    counter: dict[str, int] = {}
    for r in results:
        for f in r.findings:
            counter[f.code] = counter.get(f.code, 0) + 1
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))
