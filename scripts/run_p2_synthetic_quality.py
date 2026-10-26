#!/usr/bin/env python
"""P2（数据价值分析）· 第一部分：合成缺陷可检性 + A/B 一致性诊断 + 长尾真实覆盖。

1. 在 train 剔除 holdout 的池上，检查为何随机采样 A 与质量过滤 B 完全一致
   （根因：clean_manifest 已是 P1 质量门后的产出，附加过滤近乎空操作）。
2. 构造可核验的合成缺陷（缺失图注 / 文本异常 / 重复图像），跑真实质量算子，
   测量识别率；并验证"清洁样本"不被误杀（无假阳性）。
3. 改进长尾指标：不再只用"总 CUI 数"，而是计算**罕见（singleton）CUI 的真实覆盖**。
4. ID 错配：说明当前算子不校验跨模态 ID 一致性（ROCOv2 由构造保证），
   真实错配表现为检索任务中的 risk 案例，交 P3 人工复核，不作"自动确诊"。
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medalign.flywheel.selection import run_all_strategies  # noqa: E402
from medalign.pipeline.runner import QualityPipeline  # noqa: E402


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    root = Path(".")
    manifest = load_jsonl(root / "data/metadata/clean_manifest.jsonl")
    raw = load_jsonl(root / "data/metadata/raw_manifest.jsonl")
    raw_by_id = {r["sample_id"]: r for r in raw}

    out: dict = {"synthetic_defect_test": {}, "ab_diagnosis": {}, "longtail_coverage": {}}

    # ---------------- 1) A/B 一致性诊断 ----------------
    train_rows = [r for r in manifest if r.get("split") == "train"]
    n_all = len(train_rows)
    n_quality_ok = sum(
        1
        for r in train_rows
        if r.get("decision") == "keep"
        and not (
            set(r.get("warn_codes") or [])
            & {"CAPTION_HIGH_DIGIT", "CAPTION_REPETITIVE", "IMAGE_LOW_CONTRAST", "IMAGE_ASPECT_SKEW"}
        )
    )
    out["ab_diagnosis"] = {
        "n_train": n_all,
        "n_pass_quality_filter_B": n_quality_ok,
        "fraction_passing_B": round(n_quality_ok / max(1, n_all), 4),
        "root_cause": (
            "clean_manifest 本身就是 P1 质量门后的保留集合（仅 10/1000 被拒），"
            "因此策略 B 的附加过滤在已清洁子集上几乎不改变候选集；"
            "要证明质量过滤'真的有效'，必须在未清洁数据上注入缺陷来测（见下方合成测试）。"
        ),
    }
    print(f"[A/B] train={n_all} pass_B={n_quality_ok} ({out['ab_diagnosis']['fraction_passing_B']})")

    # ---------------- 2) 合成缺陷可检性 ----------------
    # 取若干真实清洁记录作为"控制样本"（应全部 keep，验证无假阳性）
    controls = [r for r in train_rows if r["sample_id"] in raw_by_id][:8]
    defect_records: list[dict] = []
    expected: dict[str, list[str]] = {}

    def make_defect(sid_suffix: str, caption, **overrides) -> dict:
        base = dict(raw_by_id[controls[0]["sample_id"]])  # 真实图像，保证图像算子可读
        base.update(overrides)
        base["sample_id"] = f"SYN_{sid_suffix}"
        base["split"] = "train"
        base["caption"] = caption
        base["row_idx"] = 999000 + len(defect_records)
        return base

    # 文本类缺陷（只需 caption 字段）
    d_missing = make_defect("missing_caption", None)
    defect_records.append(d_missing)
    expected["SYN_missing_caption"] = ["CAPTION_MISSING"]

    d_empty = make_defect("empty_caption", "")
    defect_records.append(d_empty)
    expected["SYN_empty_caption"] = ["CAPTION_EMPTY"]

    d_placeholder = make_defect("placeholder", "nan")
    defect_records.append(d_placeholder)
    expected["SYN_placeholder"] = ["CAPTION_PLACEHOLDER"]

    d_repetitive = make_defect("repetitive", "CT CT CT CT CT CT CT CT CT CT CT CT CT CT CT")
    defect_records.append(d_repetitive)
    expected["SYN_repetitive"] = ["CAPTION_REPETITIVE"]  # 警告级，也算检出

    d_digit = make_defect("digit", "123 456 789 012 345 678 901 234 567 890 111 222 333 444 555")
    defect_records.append(d_digit)
    expected["SYN_digit"] = ["CAPTION_HIGH_DIGIT"]

    # 重复图像：复制一条真实记录，换 sample_id 与图注，但复用同一真实 image_relpath
    dup_base = dict(raw_by_id[controls[1]["sample_id"]])
    dup_a = dict(dup_base); dup_a["sample_id"] = "SYN_dupimg_a"; dup_a["split"] = "train"
    dup_b = dict(dup_base); dup_b["sample_id"] = "SYN_dupimg_b"; dup_b["split"] = "train"
    dup_b["caption"] = "Axial T2 weighted image of the brain showing nonspecific white matter changes."
    defect_records.extend([dup_a, dup_b])
    dup_ids = ["SYN_dupimg_a", "SYN_dupimg_b"]

    # ID 错配 / 重复 ID：注入一个与真实样本撞 sample_id 的记录（provenance 冲突）
    id_collision = dict(raw_by_id[controls[2]["sample_id"]])
    id_collision["sample_id"] = controls[2]["sample_id"]  # 故意重复真实 id
    id_collision["split"] = "train"
    id_collision["caption"] = "This caption intentionally reuses an existing sample_id to test ID consistency."
    defect_records.append(id_collision)
    expected["ID_COLLISION"] = []  # 当前算子无 ID 唯一性校验 → 预期无自动检出（如实记录）

    # 写入临时 raw 清单并跑真实管线（输出到隔离目录，绝不覆盖真实 clean_manifest）
    aug = controls + defect_records
    tmp_dir = root / "data/metadata/_synthetic"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    aug_path = tmp_dir / "raw_augmented.jsonl"
    aug_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in aug), encoding="utf-8")

    pipe = QualityPipeline(
        raw_root="data/raw", meta_dir=str(tmp_dir),
        image_thresholds={}, text_thresholds={}, dedup_thresholds={},
    )
    pipe.run(aug_path)

    # 读取合成结果
    syn_clean = load_jsonl(tmp_dir / "clean_manifest.jsonl")
    syn_rejected = load_jsonl(tmp_dir / "rejected_manifest.jsonl")
    by_id = {r["sample_id"]: r for r in syn_clean + syn_rejected}
    clusters = json.loads((tmp_dir / "duplicate_clusters.json").read_text(encoding="utf-8"))

    # 控制样本应全部 keep（无假阳性）
    control_results = {}
    for c in controls:
        r = by_id.get(c["sample_id"], {})
        control_results[c["sample_id"]] = r.get("decision")
    n_control_keep = sum(1 for v in control_results.values() if v == "keep")

    # 缺陷检出
    detection: dict[str, dict] = {}
    for sid, exp in expected.items():
        r = by_id.get(sid, {})
        if sid == "ID_COLLISION":
            detection[sid] = {"expected": "none", "decision": r.get("decision"),
                              "codes": r.get("warn_codes", []) + r.get("reject_codes", []),
                              "auto_detected": False,
                              "note": "当前算子无跨模态/ID 唯一性校验；预期无自动检出（如实记录为已知缺口）"}
            continue
        codes = r.get("reject_codes", []) + r.get("warn_codes", [])
        detected = any(e in codes for e in exp)
        detection[sid] = {"expected_codes": exp, "decision": r.get("decision"),
                          "codes": codes, "auto_detected": detected}

    # 重复图像：检查是否形成簇
    dup_cluster_hit = any(
        len([m for m in c["members"] if m in dup_ids]) >= 2 for c in clusters["clusters"]
    )

    out["synthetic_defect_test"] = {
        "n_controls": len(controls),
        "n_controls_kept_no_false_positive": n_control_keep,
        "n_defects_injected": len([d for d in defect_records if d["sample_id"].startswith("SYN_")]),
        "detection_per_defect": detection,
        "duplicate_image_cluster_detected": dup_cluster_hit,
        "id_collision_auto_detected": detection["ID_COLLISION"]["auto_detected"],
        "summary": {
            "auto_detectable_defects": sum(1 for k, v in detection.items() if k != "ID_COLLISION" and v["auto_detected"]),
            "auto_detectable_total": sum(1 for k in detection if k != "ID_COLLISION"),
            "false_positive_on_clean_controls": len(controls) - n_control_keep,
        },
    }
    print(f"[SYN] controls_kept={n_control_keep}/{len(controls)} "
          f"defects_auto_detected={out['synthetic_defect_test']['summary']['auto_detectable_defects']}/"
          f"{out['synthetic_defect_test']['summary']['auto_detectable_total']} "
          f"dup_img_cluster={dup_cluster_hit}")

    # ---------------- 3) 长尾真实覆盖（singleton CUI） ----------------
    # 在 train 剔除 holdout 的池上比较 随机 A 与 长尾 D 的"罕见概念覆盖"
    holdout_ids = set()
    hp = root / "data/metadata/holdout_sealed.jsonl"
    if hp.exists():
        for l in hp.read_text(encoding="utf-8").splitlines():
            if l.strip():
                holdout_ids.add(json.loads(l)["sample_id"])
    pool_rows = [r for r in train_rows if r["sample_id"] not in holdout_ids]
    # 罕见概念 = 仅在 1 条样本出现的 CUI（singleton）
    cui_freq = Counter()
    for r in pool_rows:
        cui_freq.update(set(r.get("cui") or []))
    singleton_cuis = {c for c, n in cui_freq.items() if n == 1}
    n_singleton = len(singleton_cuis)
    n_total_cui = len(cui_freq)

    def coverage(rows) -> dict:
        sel = set(rows)
        sel_cui = Counter()
        for r in pool_rows:
            if r["sample_id"] in sel:
                sel_cui.update(set(r.get("cui") or []))
        cov_singleton = len(singleton_cuis & set(sel_cui)) / max(1, n_singleton)
        cov_total = len(set(sel_cui)) / max(1, n_total_cui)
        return {
            "n_selected": len(sel),
            "distinct_cui": len(set(sel_cui)),
            "singleton_cui_covered": len(singleton_cuis & set(sel_cui)),
            "singleton_cui_coverage": round(cov_singleton, 4),
            "total_cui_coverage": round(cov_total, 4),
        }

    resA = run_all_strategies(pool_rows, 200, 20260701, "train", hard_ids=set(), near_dup_ids=set())
    covA = coverage(resA["A_random"].sample_ids)
    covD = coverage(resD := resA["D_quality_longtail"].sample_ids)
    out["longtail_coverage"] = {
        "n_singleton_cui_in_pool": n_singleton,
        "n_total_cui_in_pool": n_total_cui,
        "strategy_A_random": covA,
        "strategy_D_longtail": covD,
        "note": ("D 优先稀缺 CUI，理论上应提升 singleton 覆盖；若二者接近，说明 200 预算下"
                 "随机已覆盖大部分罕见概念，长尾增益有限——如实报告，不夸大。"),
    }
    print(f"[LONGTAIL] singleton cov A={covA['singleton_cui_coverage']} D={covD['singleton_cui_coverage']} "
          f"| total cov A={covA['total_cui_coverage']} D={covD['total_cui_coverage']}")

    # 清理隔离目录
    shutil.rmtree(tmp_dir, ignore_errors=True)

    Path("reports").mkdir(exist_ok=True)
    Path("reports/synthetic_quality.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[saved] reports/synthetic_quality.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
