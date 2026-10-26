#!/usr/bin/env python
"""P2（数据价值分析）· 第三部分：229 条难例分类。

四类（主类别，按信号归属，允许重叠后在 by_flag 中报告）：
  1) model_ranking_error  —— risk_false_positive：模型把某负例排得比正例更靠前（确实排序错误）
  2) borderline_positive   —— hard_positive：正例未排第 1 但仍在结果中（边界样本）
  3) caption_ambiguity     —— 含共享概念 / 歧义被排除：图文配对本身存在歧义
  4) data_quality_suspect  —— 样本在 clean_manifest 带质量告警/拒绝码（疑似数据问题）

贯穿属性：
  - needs_human_review：所有难例都是复核候选（由挖掘定义决定），单列统计。
  - ambiguity_excluded：被歧义规则显式排除的配对，必须人工确认后才可进入训练。

重要边界：未人工审核的 risk_false_positive 仅表示"模型检索排序把高相似负例置于正例之前"，
不等同于"已确认的医学错配"；真正的错配须经人工复核字段确认，不得冒充确定错误。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

sys_path_patch = Path(__file__).resolve().parents[1] / "src"


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    root = Path(".")
    hard = load_jsonl(root / "reports/hard_cases.jsonl")
    manifest = {r["sample_id"]: r for r in load_jsonl(root / "data/metadata/clean_manifest.jsonl")}

    quality_flag_codes = {
        "CAPTION_MISSING", "CAPTION_EMPTY", "CAPTION_TOO_SHORT", "CAPTION_PLACEHOLDER",
        "CAPTION_HIGH_DIGIT", "CAPTION_REPETITIVE", "IMAGE_LOW_CONTRAST", "IMAGE_ASPECT_SKEW",
        "IMAGE_SMALL", "DUP_NEAR_IMAGE", "DUP_EXACT_BYTES", "CONCEPT_EMPTY",
    }

    by_primary = Counter()
    by_flag = Counter()
    for h in hard:
        sid = h["sample_id"]
        codes = set(manifest.get(sid, {}).get("warn_codes") or []) | set(
            manifest.get(sid, {}).get("reject_codes") or []
        )
        ambiguity_excluded = bool(h.get("ambiguity_excluded", False))
        shared = h.get("concept_group") == "has_shared_concept"
        kind = h.get("kind")

        # 主类别：先数据质量，再按挖掘 kind 区分（歧义为普遍属性，单列 flag 不抢主类别）
        if codes & quality_flag_codes:
            primary = "data_quality_suspect"
        elif kind == "risk_false_positive":
            primary = "model_ranking_error"
        elif kind == "hard_positive":
            primary = "borderline_positive"
        else:
            primary = "caption_ambiguity"
        by_primary[primary] += 1
        h["_taxonomy_primary"] = primary

        # 贯穿标志
        by_flag["needs_human_review"] += 1
        if ambiguity_excluded:
            by_flag["ambiguity_excluded"] += 1
        if shared:
            by_flag["shared_concept"] += 1
        if codes & quality_flag_codes:
            by_flag["has_quality_code"] += 1

    out = {
        "n_total": len(hard),
        "by_primary": dict(by_primary),
        "by_flag": dict(by_flag),
        "caveat": (
            "risk_false_positive 仅表示模型检索排序把高相似负例置于正例之前，"
            "不等于已确认的医学错配；真正错配须经人工复核字段确认，不得冒充确定错误。"
        ),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/hard_case_taxonomy.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[taxonomy primary]", json.dumps(out["by_primary"], ensure_ascii=False))
    print("[taxonomy flag]", json.dumps(out["by_flag"], ensure_ascii=False))
    print("[saved] reports/hard_case_taxonomy.json")
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(sys_path_patch))
    raise SystemExit(main())
