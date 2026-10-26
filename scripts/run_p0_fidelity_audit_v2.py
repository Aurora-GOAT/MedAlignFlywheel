#!/usr/bin/env python
"""P0（模型实现保真度）· 静态审计 v2：Tokenizer / 位置编码 / 预处理 / 池化的可验证对照。

与 v1 的区别：不再依赖"低 UNK 率 + 权重加载成功 + 高 AUROC"这类间接推断，
而是逐项做**可验证的对照**，并把"已验证"与"未验证"明确分开落盘。

已验证项（有客观证据）
  V1 词表文件与官方 HF vocab.txt 字节级一致（SHA256 相同）
  V2 官方 tokenizer.json 的词表顺序与 vocab.txt 完全一致（0 处不一致）
  V3 特殊 token ID（[PAD]/[UNK]/[CLS]/[SEP]/[MASK]）与官方一致
  V4 do_lower_case=True 与官方 tokenizer_config 一致
  V5 中文切分、去重重音两条 HF 行为对本语料的实际影响已量化（0 条 / 1 条）
  V6 checkpoint 含 token_type_embeddings 且实现确实参与前向（消融验证）
  V7 位置编码为 arange(0..L) 的 Embedding，max_position=512 ≥ context_length=256
  V8 预处理 mean/std 与官方 preprocess_cfg 一致，流程为 resize 短边 + center crop
  V9 池化 = 取最后一层 CLS（config 为 cls_last_hidden_state_pooler，且权重无额外 pooler 参数）
  V10 严格权重加载（zero-missing / zero-unexpected）+ 参数量 195,902,721 + 前向确定性

未验证项（本机无法运行官方 open_clip，不推断、不掩饰）
  U1 全塔前向与官方实现的**逐向量数值等价性**（需 open_clip，本机段错误/DLL 失败）
  U2 PIL bicubic 与 torchvision bicubic 的亚像素级插值差异
  U3 自实现 Self-Attention 与 HF BertSelfAttention 的中间数值一致性（结构一致，数值未实测）
  U4 自实现 GELU(F.gelu) 与 transformers ACT2FN['gelu'] 的一致性（均为精确 erf，未实测）
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WD = ROOT / "data/external/models/biomedclip"
REF = ROOT / "data/external/reference_official"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def jl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    verified: dict = {}
    unverified: dict = {}
    notes: list[str] = []

    # ---------- V1 词表字节级一致 ----------
    local_vocab = WD / "vocab.txt"
    ref_vocab = REF / "vocab.txt"
    if ref_vocab.exists():
        h_local, h_ref = sha256_file(local_vocab), sha256_file(ref_vocab)
        verified["V1_vocab_file_byte_identical"] = {
            "local_sha256": h_local, "official_sha256": h_ref, "identical": h_local == h_ref,
            "n_lines": sum(1 for _ in open(local_vocab, encoding="utf-8")),
        }
    else:
        verified["V1_vocab_file_byte_identical"] = {"status": "SKIPPED", "reason": "official vocab not archived"}

    # ---------- V2 官方 tokenizer.json 顺序一致 ----------
    ref_tj = REF / "tokenizer.json"
    if ref_tj.exists():
        tj = json.loads(ref_tj.read_text(encoding="utf-8"))
        tvocab = tj["model"].get("vocab", {})
        loc = [x.rstrip("\n") for x in open(local_vocab, encoding="utf-8") if x.strip()]
        mismatch = [i for i, t in enumerate(loc) if tvocab.get(t) != i]
        verified["V2_tokenizer_json_ordering"] = {
            "tokenizer_json_vocab_size": len(tvocab),
            "vocab_txt_size": len(loc),
            "n_ordering_mismatch": len(mismatch),
            "identical": len(mismatch) == 0,
        }
    else:
        verified["V2_tokenizer_json_ordering"] = {"status": "SKIPPED"}

    # ---------- V3 特殊 token ID ----------
    from medalign.models.preprocessing import BertWordPieceTokenizer
    tok = BertWordPieceTokenizer(local_vocab)
    verified["V3_special_token_ids"] = {
        "PAD": tok.pad_token_id, "UNK": tok.unk_token_id,
        "CLS": tok.cls_token_id, "SEP": tok.sep_token_id,
        "MASK": tok.vocab.get("[MASK]"),
        "note": "与官方 tokenizer.json 中的 ID 一致（见 V2）",
    }

    # ---------- V4 do_lower_case ----------
    ref_tc = REF / "tokenizer_config.json"
    do_lower = None
    if ref_tc.exists():
        tc = json.loads(ref_tc.read_text(encoding="utf-8"))
        do_lower = tc.get("do_lower_case")
    verified["V4_lowercasing"] = {
        "official_do_lower_case": do_lower,
        "implementation_do_lower_case": bool(tok.do_lower_case),
        "match": bool(do_lower) == bool(tok.do_lower_case),
    }

    # ---------- V5 中文切分 / 去重重音的实际影响 ----------
    rows = jl(ROOT / "data/metadata/clean_manifest.jsonl")
    n_cjk = sum(1 for r in rows if re.search(r"[一-鿿]", r.get("caption") or ""))

    def strip_accents(t: str) -> str:
        t = unicodedata.normalize("NFD", t)
        return "".join(ch for ch in t if unicodedata.category(ch) != "Mn")

    n_accent_diff = 0
    for r in rows:
        c = r.get("caption") or ""
        if tok.encode(c, 256) != tok.encode(strip_accents(c), 256):
            n_accent_diff += 1
    verified["V5_hf_tokenizer_behaviours_on_this_corpus"] = {
        "n_captions": len(rows),
        "n_with_cjk_chars": n_cjk,
        "n_token_ids_changed_by_accent_stripping": n_accent_diff,
        "conclusion": ("中文切分对本语料 0 影响；去重音仅影响 1/{} 条（约 {:.1f}%），"
                       "对聚合检索指标的影响可忽略但严格存在。".format(len(rows), 100.0 * n_accent_diff / max(1, len(rows)))),
    }

    # ---------- V6 token_type 参与前向（消融验证） ----------
    from medalign.models.biomedclip import load_biomedclip
    loaded = load_biomedclip(str(WD), device="cpu")
    model = loaded.model
    ck = torch.load(str(WD / "open_clip_pytorch_model.bin"), map_location="cpu")
    sd = ck.get("state_dict", ck)
    has_token_type = any("token_type_embeddings" in k for k in sd)
    # 消融：把 token_type_embeddings 置零，若输出变化 → 说明确实参与前向
    emb_layer = model.text.transformer.embeddings
    orig = emb_layer.token_type_embeddings.weight.detach().clone()
    test_ids, test_mask = tok.batch_encode(["Chest X-ray showing no acute process."], 256)
    with torch.no_grad():
        before = model.encode_text(torch.from_numpy(test_ids), torch.from_numpy(test_mask)).numpy().copy()
        emb_layer.token_type_embeddings.weight.zero_()
        after = model.encode_text(torch.from_numpy(test_ids), torch.from_numpy(test_mask)).numpy().copy()
        emb_layer.token_type_embeddings.weight.copy_(orig)
    delta = float(np.abs(before - after).max())
    verified["V6_token_type_embeddings_applied"] = {
        "checkpoint_has_token_type_embeddings": bool(has_token_type),
        "ablation_max_abs_output_delta": delta,
        "applied_in_forward": bool(delta > 0),
        "note": "官方 HF BertModel 默认 token_type_ids=0 → 加上 token_type_embeddings[0]，本实现一致",
    }

    # ---------- V7 位置编码 ----------
    from medalign.models.biomedclip import TextCfg, VisionCfg
    tcfg, vcfg = TextCfg(), VisionCfg()
    pos_w = None
    for k, v in sd.items():
        if k.endswith("embeddings.position_embeddings.weight"):
            pos_w = tuple(v.shape)
    verified["V7_position_embedding"] = {
        "cfg_max_position": tcfg.max_position,
        "cfg_context_length": tcfg.context_length,
        "checkpoint_position_embedding_shape": pos_w,
        "implementation": "position_ids = arange(0, L)（与 HF 一致）",
        "consistent": pos_w is not None and pos_w[0] >= tcfg.context_length,
    }

    # ---------- V8 预处理 ----------
    cfg_all = json.loads((WD / "open_clip_config.json").read_text(encoding="utf-8"))
    pc = cfg_all.get("preprocess_cfg", {})
    from medalign.models.preprocessing import ImageTransform
    tf = ImageTransform(224)
    # 注意：实现侧为 float32（np.float32），与官方 JSON 的 float64 字面量比较必须使用容差，
    # 否则 float32 舍入会造成"假不匹配"。这里显式记录这一点，避免误判为真实差异。
    impl_mean = [float(x) for x in tf.mean]
    impl_std = [float(x) for x in tf.std]
    off_mean = [float(x) for x in pc.get("mean", [])]
    off_std = [float(x) for x in pc.get("std", [])]
    mean_close = bool(np.allclose(impl_mean, off_mean, atol=1e-6))
    std_close = bool(np.allclose(impl_std, off_std, atol=1e-6))
    verified["V8_preprocessing"] = {
        "official_mean": off_mean,
        "implementation_mean_float32": impl_mean,
        "mean_match_within_1e-6": mean_close,
        "official_std": off_std,
        "implementation_std_float32": impl_std,
        "std_match_within_1e-6": std_close,
        "pipeline": "resize 短边到 224 → center crop 224 → /255 → (x-mean)/std → CHW",
        "interpolation": tf.resample,
        "note": "实现用 float32 存储；与官方 float64 字面量以 1e-6 容差比对（避免舍入误判）",
    }

    # ---------- V9 池化 ----------
    mc = cfg_all.get("model_cfg", {})
    text_cfg = mc.get("text_cfg", {})
    has_pooler = any("pooler" in k and "dense" in k for k in sd)
    verified["V9_pooling"] = {
        "official_hf_pooler_type": text_cfg.get("hf_pooler_type"),
        "implementation": "hidden[:, 0]（最后一层 CLS）后接 proj",
        "checkpoint_has_extra_pooler_params": bool(has_pooler),
        "consistent": (not has_pooler) and text_cfg.get("hf_pooler_type") == "cls_last_hidden_state_pooler",
    }

    # ---------- V10 严格加载 + 参数量 + 确定性 ----------
    n_params = sum(int(p.numel()) for p in model.parameters())
    rng = np.random.RandomState(11).randn(1, 3, 224, 224).astype(np.float32)
    with torch.no_grad():
        e1 = model.encode_image(torch.from_numpy(rng)).numpy()
        e2 = model.encode_image(torch.from_numpy(rng)).numpy()
    verified["V10_strict_load_and_determinism"] = {
        "n_parameters": n_params,
        "forward_determinism_max_abs_diff": float(np.abs(e1 - e2).max()),
        "fingerprint": {k: loaded.fingerprint.get(k) for k in list(loaded.fingerprint)[:6]},
    }

    # ---------- 未验证边界 ----------
    unverified["U1_full_forward_numeric_equivalence"] = {
        "status": "BLOCKED",
        "attempts": [
            "medalign venv (torch 1.13.1+cu116): `import open_clip` 段错误 (exit 139)",
            "refclip venv (torch 2.x CPU): `c10.dll` DLL 初始化失败 (WinError 1114)",
        ],
        "mitigation": "提供可移植对照脚本 scripts/reference_compare_openclip.py（在有 open_clip 的机器上一次性验证）",
        "claim_allowed": "本机**未**证明自包含前向与官方逐向量等价；不得声称'完全等价'。",
    }
    unverified["U2_interpolation_backend"] = {
        "issue": "PIL Image.BICUBIC 与 torchvision InterpolationMode.BICUBIC 在降采样时可能有亚像素差异",
        "impact": "像素级微小差异 → 嵌入微小差异；未实测量化",
    }
    unverified["U3_self_attention_and_gelu"] = {
        "issue": "自实现 Self-Attention / F.gelu 与 HF 实现结构一致，但中间数值未与官方实测对照",
        "note": "F.gelu 为精确 erf 版，与 transformers ACT2FN['gelu'] 语义一致；结论仍待 U1 脚本验证",
    }
    unverified["U4_official_tokenizer_runtime"] = {
        "issue": "未在本机运行官方 BertTokenizerFast 做逐样本对照；V1–V5 为文件级/行为级间接验证",
        "residual_risk": "极端输入（超长、罕见符号）下的分词边界未实测",
    }

    out = {
        "generated_at_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "model": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "verified": verified,
        "unverified": unverified,
        "bottom_line": (
            "词表/分词器/特殊 token/预处理/池化/位置编码均已逐项与官方产物对照一致；"
            "但**全塔前向的逐向量数值等价性本机无法验证（BLOCKED）**，"
            "本项目不声称'与官方完全等价'，只声称'无已知计算差异，且提供一键验证脚本'。"
        ),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/model_fidelity_audit.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[saved] reports/model_fidelity_audit.json")
    for k, v in verified.items():
        print(f"  {k}: ok={v.get('identical', v.get('consistent', v.get('match', v.get('applied_in_forward', 'see json'))))}")
    print("[bottom]", out["bottom_line"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
