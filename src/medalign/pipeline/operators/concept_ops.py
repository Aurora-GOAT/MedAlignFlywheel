"""医学概念（UMLS CUI）质量算子。

重要前提：HF 版本的 ROCOv2 只给出 CUI 列表，**没有概念名称，也没有语义类型**。
本项目没有 UMLS 授权，因此无法做"概念是否为疾病/操作/解剖部位"的语义校验，
更无法判断"这个概念和这张图是否真的相关"。

本算子只做能做且可核查的部分：
- CUI 字面格式是否合法；
- 是否重复；
- 是否为空一轮覆盖缺失；
- 在整个候选集里的出现频次（长尾 / 罕见概念识别，用于后续数据选择）。

任何输出的"概念质量"结论都必须在报告里标注为
**基于 CUI 分布的代理指标**，而不是人工医学判定。
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from ..operators.base import OK, WARN, Finding, OpResult, Operator

OP_VERSION = "concept-cui-v1.0"

CUI_RE = re.compile(r"^C\d{7}$")


class ConceptQualityOperator(Operator):
    name = "concept_quality"
    version = OP_VERSION
    proxy_only = True

    def __init__(self, global_cui_freq: Counter | None = None, min_support: int = 1):
        self.freq = global_cui_freq or Counter()
        self.min_support = min_support

    @staticmethod
    def build_freq(records: Iterable[dict]) -> Counter:
        freq: Counter = Counter()
        for r in records:
            for c in set(r.get("cui") or []):
                freq[c] += 1
        return freq

    def run(self, record: dict, ctx: dict) -> OpResult:
        sid = record["sample_id"]
        res = OpResult(sample_id=sid)
        cuis = list(record.get("cui") or [])
        res.metrics["n_cui_raw"] = len(cuis)

        if not cuis:
            res.findings.append(
                Finding(self.name, self.version, WARN, "CONCEPT_EMPTY",
                        "该样本没有任何 CUI 标注",
                        evidence={"note": "不影响图文配对，但无法参与概念层面的分析"})
            )
            res.metrics["n_cui_valid"] = 0
            res.metrics["cui_min_support"] = None
            return res

        valid = [c for c in cuis if CUI_RE.match(str(c))]
        invalid = [c for c in cuis if not CUI_RE.match(str(c))]
        unique = sorted(set(valid))

        res.metrics["n_cui_valid"] = len(valid)
        res.metrics["n_cui_unique"] = len(unique)
        res.metrics["cui_list"] = unique

        if invalid:
            res.findings.append(
                Finding(self.name, self.version, WARN, "CONCEPT_INVALID_FORMAT",
                        "存在不符合 C####### 格式的 CUI",
                        evidence={"invalid": invalid[:5]}, threshold="^C\\d{7}$")
            )

        if len(valid) != len(unique):
            res.findings.append(
                Finding(self.name, self.version, WARN, "CONCEPT_DUPLICATE",
                        "同一 CUI 重复出现多次",
                        evidence={"raw_count": len(valid), "unique_count": len(unique)})
            )

        supports = [self.freq.get(c, 0) for c in unique]
        if supports:
            res.metrics["cui_min_support"] = min(supports)
            res.metrics["cui_max_support"] = max(supports)
            rare = [c for c in unique if self.freq.get(c, 0) <= 1]
            res.metrics["n_cui_singleton"] = len(rare)
            res.metrics["cui_singleton_list"] = rare
            # 全部 CUI 都是独例 → 该样本的"概念覆盖"在本候选集里是长尾
            if rare and len(rare) == len(unique):
                res.findings.append(
                    Finding(self.name, self.version, WARN, "CONCEPT_ALL_LONGETAIL",
                            "该样本所有 CUI 在本候选集中都只出现一次（长尾概念）",
                            evidence={"cui": rare[:5]})
                )

        if not res.warnings:
            res.findings.append(
                Finding(self.name, self.version, OK, "CONCEPT_OK",
                        "CUI 格式与去重检查通过",
                        evidence={"n_unique": len(unique)})
            )
        return res
