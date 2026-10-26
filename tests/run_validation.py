#!/usr/bin/env python
"""MedAlignFlywheel 自动化验证测试（无外部依赖，纯标准库 + 项目源码）。

覆盖：评测链路 bug 修复（AUROC 方向 / recall@k）、P1 对齐头 step-0 恒等、
梯度数值、P1 质量算子、bootstrap 置信区间、模型前向确定性、严格权重加载。
运行：python tests/run_validation.py  （在 medalign venv 下）
退出码 0 = 全部通过，1 = 存在失败。
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

PASS = 0
FAIL = 0
LOG: list[str] = []


def load_jsonl(p: Path) -> list[dict]:
    """读取 jsonl；文件不存在时返回空列表（便于在数据未就绪时跳过）。"""
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        LOG.append(f"  PASS  {name}  {detail}")
    else:
        FAIL += 1
        LOG.append(f"  FAIL  {name}  {detail}")


def section(title: str):
    LOG.append("")
    LOG.append(f"== {title} ==")


def main() -> int:
    from medalign.common import stats  # noqa: E402
    from medalign.eval import retrieval  # noqa: E402
    from medalign.flywheel import adapter  # noqa: E402
    from medalign.pipeline.operators.text_ops import CaptionQualityOperator  # noqa: E402
    from medalign.models import biomedclip  # noqa: E402

    # ---------- 1) 评测链路 bug 修复验证 ----------
    section("1) 评测链路修复 (AUROC / recall@k)")
    # AUROC 方向：正例明显 > 负例应接近 1.0（验证计数方向修复）
    auroc_hi = stats.auroc_rank_based(np.array([0.9, 0.8, 0.85]), np.array([0.1, 0.2, 0.15]))
    check("auroc_pos_gt_neg~1", abs(auroc_hi - 1.0) < 1e-9, f"auroc={auroc_hi:.4f}")
    # AUROC 方向：正例明显 < 负例应接近 0.0（原 bug 会返回 1.0）
    auroc_lo = stats.auroc_rank_based(np.array([0.1, 0.2]), np.array([0.9, 0.8]))
    check("auroc_pos_lt_neg~0", abs(auroc_lo - 0.0) < 1e-9, f"auroc={auroc_lo:.4f}")

    # recall_at_k 语义（浮点版返回比率）
    ranks = np.array([1, 3, 6, 2])
    r1 = retrieval.recall_at_k(ranks, 1)
    r3 = retrieval.recall_at_k(ranks, 3)
    check("recall@1=0.25", abs(r1 - 0.25) < 1e-9, f"={r1}")
    check("recall@3=0.75", abs(r3 - 0.75) < 1e-9, f"={r3}")

    # bootstrap CI 应覆盖点估计
    sim = np.random.RandomState(0).randn(200, 200)
    np.fill_diagonal(sim, 5.0)
    res, i2t_ranks, _ = retrieval.compute_recall_metrics(sim, (1, 5))
    boot = retrieval.bootstrap_recall(i2t_ranks, (1, 5), n_boot=200, seed=0)
    ci = boot["recall@1"]
    check("bootstrap_ci_covers_point", ci["lo"] <= ci["point"] <= ci["hi"],
          f"[{ci['lo']:.3f},{ci['hi']:.3f}] pt={ci['point']:.3f}")

    # ---------- 2) P1 对齐头 step-0 恒等 + 梯度数值 ----------
    section("2) P1 对齐头 (step-0 恒等 / 梯度数值)")
    rng = np.random.RandomState(1)
    img = rng.randn(8, 512).astype(np.float32)
    txt = rng.randn(8, 512).astype(np.float32)
    try:
        import torch  # noqa: E402
        head = adapter.AlignmentHead(dim=512, hidden=256)
        head.eval()
        with torch.no_grad():
            zi, zt = head(torch.from_numpy(img), torch.from_numpy(txt))
        # 对齐头在 init 时输出 = 归一化(输入)；需与归一化后的输入比较
        img_n = img / np.linalg.norm(img, axis=1, keepdims=True)
        txt_n = txt / np.linalg.norm(txt, axis=1, keepdims=True)
        diff_i = float(np.abs(zi.numpy() - img_n).max())
        diff_t = float(np.abs(zt.numpy() - txt_n).max())
        check("adapter_step0_image_identity", diff_i < 1e-5, f"maxdiff={diff_i:.2e}")
        check("adapter_step0_text_identity", diff_t < 1e-5, f"maxdiff={diff_t:.2e}")
        # 梯度数值：构造一步前向，检查 loss 有限、梯度非零
        head2 = adapter.AlignmentHead(dim=512, hidden=256)
        ti = torch.from_numpy(adapter._l2(img)).requires_grad_(False)
        tt = torch.from_numpy(adapter._l2(txt))
        zt2 = head2.txt(tt)
        logits = (head2.img(ti) + ti) @ zt2.T
        target = torch.arange(8)
        loss = 0.5 * (torch.nn.functional.cross_entropy(logits, target) +
                      torch.nn.functional.cross_entropy(logits.T, target))
        loss.backward()
        grads = [p.grad for p in head2.parameters() if p.grad is not None]
        all_finite = all(torch.isfinite(g).all() for g in grads)
        any_nonzero = any((g != 0).any() for g in grads)
        check("adapter_loss_finite", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")
        check("adapter_grads_finite", all_finite, "")
        check("adapter_grads_nonzero", any_nonzero, "")
    except Exception as e:  # pragma: no cover
        check("adapter_blocked", False, f"exception: {e}")
        LOG.append(traceback.format_exc())

    # ---------- 3) P1 质量算子 ----------
    section("3) P1 质量算子 (文本)")
    op = CaptionQualityOperator(thresholds={}, duplicate_index={})

    def codes_of(record):
        res = op.run(record, {"raw_root": str(ROOT)})
        return [f.code for f in res.findings]

    # 缺失
    codes_missing = codes_of({"sample_id": "T_missing", "caption": None, "metrics": {}})
    check("caption_missing_detected", "CAPTION_MISSING" in codes_missing, str(codes_missing))
    # 占位 nan
    codes_ph = codes_of({"sample_id": "T_ph", "caption": "nan", "metrics": {}})
    check("caption_placeholder_detected", "CAPTION_PLACEHOLDER" in codes_ph, str(codes_ph))
    # 正常
    codes_ok = codes_of({"sample_id": "T_ok",
                         "caption": "Chest X-ray showing no acute cardiopulmonary process.",
                         "metrics": {}})
    defect_codes = {"CAPTION_MISSING", "CAPTION_EMPTY", "CAPTION_PLACEHOLDER", "CAPTION_TOO_SHORT",
                    "CAPTION_REPETITIVE", "CAPTION_NONPRINTABLE"}
    has_defect = any(c in defect_codes for c in codes_ok)
    check("caption_clean_no_false_positive", ("CAPTION_OK" in codes_ok) and not has_defect, str(codes_ok))

    # ---------- 4) 模型严格加载 + 前向确定性 ----------
    section("4) BiomedCLIP 严格加载 / 前向确定性 (CPU)")
    wdir = ROOT / "data/external/models/biomedclip"
    if wdir.exists():
        try:
            model = biomedclip.load_biomedclip(str(wdir), device="cpu")
            n_params = sum(int(p.numel()) for p in model.model.parameters())
            check("model_strict_load_params>1e8", n_params > 1e8, f"n={n_params:,}")
            import torch  # noqa: E402
            rng2 = np.random.RandomState(7).randn(1, 3, 224, 224).astype(np.float32)
            t_img = torch.from_numpy(rng2)
            with torch.no_grad():
                e1 = model.model.encode_image(t_img)
                e2 = model.model.encode_image(t_img)
            det = float((e1 - e2).abs().max())
            check("model_forward_deterministic", det < 1e-5, f"maxdiff={det:.2e}")
        except Exception as e:  # pragma: no cover
            check("model_load_blocked", False, f"exception: {e}")
            LOG.append(traceback.format_exc())
    else:
        LOG.append(f"  SKIP  model weights not found at {wdir}")

    # ---------- 5) 审计不变量（第二轮） ----------
    section("5) 审计不变量 (词表/特殊token/独立测试集/冻结配置)")
    # 5.1 词表与归档的官方 vocab 字节级一致
    ref_vocab = ROOT / "data/external/reference_official/vocab.txt"
    loc_vocab = ROOT / "data/external/models/biomedclip/vocab.txt"
    if ref_vocab.exists():
        import hashlib

        def _sha(p):
            return hashlib.sha256(Path(p).read_bytes()).hexdigest()
        check("vocab_byte_identical_to_official", _sha(loc_vocab) == _sha(ref_vocab),
              _sha(loc_vocab)[:16])
    else:
        LOG.append("  SKIP  official vocab not archived")

    # 5.2 特殊 token ID
    from medalign.models.preprocessing import BertWordPieceTokenizer as _Tok  # noqa: E402
    _tk = _Tok(loc_vocab)
    check("special_token_ids_match_official",
          (_tk.pad_token_id == 0 and _tk.unk_token_id == 1
           and _tk.cls_token_id == 2 and _tk.sep_token_id == 3),
          f"PAD={_tk.pad_token_id} UNK={_tk.unk_token_id} CLS={_tk.cls_token_id} SEP={_tk.sep_token_id}")

    # 5.3 独立测试集与开发集零重叠（image_id 级）
    dev_ids = {r["sample_id"] for r in load_jsonl(ROOT / "data/metadata/clean_manifest.jsonl")}
    ind_path = ROOT / "data/metadata/holdout_v2_manifest.jsonl"
    if ind_path.exists():
        ind_ids = {r["sample_id"] for r in load_jsonl(ind_path)}
        check("independent_testset_zero_overlap_with_dev",
              len(ind_ids & dev_ids) == 0, f"overlap={len(ind_ids & dev_ids)} n_ind={len(ind_ids)}")
    else:
        LOG.append("  SKIP  independent manifest not present yet")

    # 5.4 旧 holdout 已被审计为受污染（应记录，不得再当未见数据）
    ho_path = ROOT / "data/metadata/holdout_sealed.jsonl"
    if ho_path.exists():
        ho_ids = {r["sample_id"] for r in load_jsonl(ho_path)}
        hard_ids = {r["sample_id"] for r in load_jsonl(ROOT / "reports/hard_cases.jsonl")}
        overlap = len(ho_ids & hard_ids)
        check("old_holdout_contamination_recorded", overlap > 0,
              f"旧 holdout 与 hard_cases 重叠 {overlap} 条 → 已废弃，不作为未见数据")

    # 5.5 冻结配置存在且与清单哈希一致
    frozen = ROOT / "configs/frozen_final_eval.json"
    if frozen.exists():
        import hashlib

        fz = json.loads(frozen.read_text(encoding="utf-8"))
        if ind_path.exists():
            mh = hashlib.sha256(ind_path.read_bytes()).hexdigest()
            check("frozen_config_manifest_hash_matches", fz.get("manifest_sha256") == mh,
                  "配置在评价前冻结，且与被测清单一致")
        else:
            check("frozen_config_exists", True, "frozen_final_eval.json 存在")
    else:
        LOG.append("  SKIP  frozen config not present yet")

    # ---------- 汇总 ----------
    LOG.append("")
    LOG.append(f"TOTAL: {PASS} passed, {FAIL} failed")
    out = ROOT / "reports/validation_test_log.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(LOG), encoding="utf-8")
    print("\n".join(LOG))
    print(f"\n[log] {out}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
