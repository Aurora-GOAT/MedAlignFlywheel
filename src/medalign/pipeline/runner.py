"""P1 质量管线编排。

输出：
- data/metadata/clean_manifest.jsonl     清洗后保留的样本（含全部指标）
- data/metadata/rejected_manifest.jsonl  被拒绝样本 + 完整拒绝原因
- data/metadata/quality_report.json      全局统计
- data/metadata/lineage.jsonl            逐样本血缘
- data/metadata/duplicate_clusters.json  近重复簇（可复核）
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..common.hashing import write_json, write_jsonl
from ..common.textnorm import caption_fingerprint, token_set
from .lineage import PIPELINE_VERSION, LineageEntry
from .modality import (
    caption_complexity,
    infer_modality,
    infer_region,
    summarize_modality,
)
from .operators.base import OK, REJECT, WARN, Operator, OpResult, count_findings
from .operators.concept_ops import ConceptQualityOperator
from .operators.dedup_ops import DuplicateCluster, DuplicateLeakageOperator, build_clusters
from .operators.image_ops import ImageIntegrityOperator
from .operators.text_ops import CaptionQualityOperator


def _read_manifest(path: Path) -> list[dict]:
    import json

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class QualityPipeline:
    def __init__(
        self,
        raw_root: str,
        meta_dir: str,
        image_thresholds: dict | None = None,
        text_thresholds: dict | None = None,
        dedup_thresholds: dict | None = None,
    ):
        self.raw_root = Path(raw_root)
        self.meta_dir = Path(meta_dir)
        self.image_thresholds = image_thresholds or {}
        self.text_thresholds = text_thresholds or {}
        self.dedup_thresholds = dedup_thresholds or {}

    # ---------------- 阶段 1：逐样本算子（不依赖全局信息） ----------------
    def stage1_per_sample(self, records: list[dict]) -> tuple[list[dict], list[OpResult]]:
        # 先做一次 caption 指纹索引（重复图注会影响检索评测的负例构造）
        dup_index: dict[str, list[str]] = defaultdict(list)
        for r in records:
            if r.get("caption"):
                dup_index[caption_fingerprint(str(r["caption"]))].append(r["sample_id"])

        cui_freq = ConceptQualityOperator.build_freq(records)

        image_op: Operator = ImageIntegrityOperator(self.image_thresholds)
        text_op: Operator = CaptionQualityOperator(self.text_thresholds, dup_index)
        concept_op: Operator = ConceptQualityOperator(cui_freq)

        ctx = {"raw_root": str(self.raw_root)}
        enriched: list[dict] = []
        results: list[OpResult] = []
        for r in records:
            out = dict(r)
            metrics: dict[str, Any] = {}
            rr = out.setdefault("metrics", {})
            for op in (image_op, text_op, concept_op):
                res = op.run(out, ctx)
                metrics[op.name] = {
                    "version": op.version,
                    "codes": [f.code for f in res.findings],
                    "severity": {f.code: f.severity for f in res.findings},
                }
                rr.update(res.metrics)
            results.append(res_stub(out, metrics))
            enriched.append(out)
        return enriched, results

    # ---------------- 阶段 2：近重复 / 泄漏（依赖全局） ----------------
    def stage2_dedup(self, enriched: list[dict]) -> tuple[list[DuplicateCluster], list[OpResult]]:
        clusters = build_clusters(enriched, self.dedup_thresholds)
        split_of = {r["sample_id"]: r.get("split") for r in enriched}
        ctx = {"split_of": split_of}
        op = DuplicateLeakageOperator(clusters)
        results = []
        for r in enriched:
            res = op.run(r, ctx)
            r.setdefault("metrics", {}).update(res.metrics)
            r["ops"]["dedup_leakage"] = {
                "version": op.version,
                "codes": [f.code for f in res.findings],
                "severity": {f.code: f.severity for f in res.findings},
            }
            results.append(res)
        return clusters, results

    # ---------------- 阶段 3：派生标签（模态/部位/复杂度代理） ----------------
    def stage3_derive(self, enriched: list[dict]) -> None:
        for r in enriched:
            cap = str(r.get("caption") or "")
            r["modality_proxy"] = infer_modality(cap)
            r["region_proxy"] = infer_region(cap)
            r["complexity"] = caption_complexity(cap)
            r["n_tokens"] = r["complexity"]["n_tokens"]
            r["tokens"] = sorted(token_set(cap))

    # ---------------- 汇总与落盘 ----------------
    def run(self, manifest_path: str | Path) -> dict:
        records = _read_manifest(Path(manifest_path))
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        enriched, stage1_results = self.stage1_per_sample(records)
        clusters, stage2_results = self.stage2_dedup(enriched)
        self.stage3_derive(enriched)

        # 合并逐样本结果
        all_ops_by_sample: dict[str, dict] = {}
        for res in stage1_results:
            all_ops_by_sample.setdefault(res.sample_id, {}).update(res.metrics)
        for res in stage2_results:
            for f in res.findings:
                all_ops_by_sample.setdefault(res.sample_id, {})

        clean_rows, rejected_rows, lineage_rows = [], [], []
        for rec, s1, s2 in zip(enriched, stage1_results, stage2_results):
            findings = s1.findings + s2.findings
            reject_codes = [f.code for f in findings if f.severity == REJECT]
            warn_codes = [f.code for f in findings if f.severity == WARN]
            decision = "reject" if reject_codes else "keep"
            rec["decision"] = decision
            rec["reject_codes"] = reject_codes
            rec["warn_codes"] = warn_codes
            rec["all_findings"] = [f.to_dict() for f in findings]

            entry = LineageEntry(
                sample_id=rec["sample_id"],
                source_dataset="ROCOv2-radiology (eltorio)",
                source_split=rec.get("split", "?"),
                source_row_idx=rec.get("row_idx"),
                upstream_image_url=rec.get("image_upstream_url", ""),
                image_sha256=rec.get("image_sha256", ""),
                operators=[
                    {"name": ImageIntegrityOperator.name, "version": ImageIntegrityOperator.version,
                     "params": self.image_thresholds},
                    {"name": CaptionQualityOperator.name, "version": CaptionQualityOperator.version,
                     "params": self.text_thresholds},
                    {"name": ConceptQualityOperator.name, "version": ConceptQualityOperator.version,
                     "params": {}},
                    {"name": DuplicateLeakageOperator.name, "version": DuplicateLeakageOperator.version,
                     "params": self.dedup_thresholds},
                ],
                decision=decision,
                reject_codes=reject_codes,
                warn_codes=warn_codes,
            )
            lineage_rows.append(entry)
            if decision == "keep":
                clean_rows.append(rec)
            else:
                rejected_rows.append(rec)

        clean_path = self.meta_dir / "clean_manifest.jsonl"
        rejected_path = self.meta_dir / "rejected_manifest.jsonl"
        lineage_path = self.meta_dir / "lineage.jsonl"
        write_jsonl(clean_path, clean_rows)
        write_jsonl(rejected_path, rejected_rows)
        with open(lineage_path, "w", encoding="utf-8", newline="\n") as f:
            for e in lineage_rows:
                f.write(e.to_json() + "\n")

        clusters_path = self.meta_dir / "duplicate_clusters.json"
        write_json(
            clusters_path,
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "pipeline_version": PIPELINE_VERSION,
                "thresholds": self.dedup_thresholds,
                "n_clusters": len(clusters),
                "clusters": [
                    {
                        "cluster_id": c.cluster_id,
                        "kind": c.kind,
                        "members": c.members,
                        "crosses_split": c.crosses_split,
                        "evidence": c.evidence,
                    }
                    for c in clusters
                ],
            },
        )

        report = self.build_report(enriched, clusters, lineage_rows)
        write_json(self.meta_dir / "quality_report.json", report)
        return report

    def build_report(self, enriched: list[dict], clusters: list[DuplicateCluster], lineage) -> dict:
        n_total = len(enriched)
        by_split = Counter(r.get("split") for r in enriched)
        keep = [r for r in enriched if r["decision"] == "keep"]
        rej = [r for r in enriched if r["decision"] == "reject"]
        reject_codes: Counter = Counter()
        warn_codes: Counter = Counter()
        for r in enriched:
            reject_codes.update(r["reject_codes"])
            warn_codes.update(r["warn_codes"])

        def split_table(rows):
            return dict(Counter(r.get("split") for r in rows))

        modality_all = summarize_modality(r["modality_proxy"]["modality"] for r in enriched)
        modality_keep = summarize_modality(r["modality_proxy"]["modality"] for r in keep)
        n_cross = sum(1 for c in clusters if c.crosses_split)

        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": PIPELINE_VERSION,
            "n_total": n_total,
            "n_keep": len(keep),
            "n_reject": len(rej),
            "keep_rate": round(len(keep) / max(1, n_total), 4),
            "by_split_total": dict(by_split),
            "by_split_keep": split_table(keep),
            "by_split_reject": split_table(rej),
            "reject_code_counts": dict(sorted(reject_codes.items(), key=lambda kv: (-kv[1], kv[0]))),
            "warn_code_counts": dict(sorted(warn_codes.items(), key=lambda kv: (-kv[1], kv[0]))),
            "duplicate_clusters": {
                "n_total": len(clusters),
                "n_cross_split": n_cross,
                "by_kind": dict(Counter(c.kind for c in clusters)),
            },
            "modality_proxy_total": modality_all,
            "modality_proxy_keep": modality_keep,
            "modality_disclaimer": (
                "模态标签由本项目规则从图注推断，ROCOv2 的 HF 版本未提供官方 modality 字段；"
                "该标签是数据质量代理指标，不是医学影像分类真值。"
            ),
            "cui_stats": {
                "n_samples_with_cui": sum(1 for r in enriched if (r.get("cui") or [])),
                "n_samples_without_cui": sum(1 for r in enriched if not (r.get("cui") or [])),
                "n_unique_cui": len(
                    {c for r in enriched for c in (r.get("cui") or [])}
                ),
                "n_singleton_cui": len(
                    {
                        c
                        for r in enriched
                        if (r.get("metrics") or {}).get("cui_singleton_list")
                        for c in r["metrics"]["cui_singleton_list"]
                    }
                ),
            },
        }


def res_stub(record: dict, metrics: dict) -> OpResult:
    """把 stage1 各算子的 Finding 汇总成一个 OpResult（便于统一后续处理）。"""
    from .operators.base import Finding

    res = OpResult(sample_id=record["sample_id"])
    for op_name, info in metrics.items():
        for code, sev in info["severity"].items():
            res.findings.append(
                Finding(op=op_name, version=info["version"], severity=sev, code=code, message="")
            )
    record["ops"] = metrics
    return res
