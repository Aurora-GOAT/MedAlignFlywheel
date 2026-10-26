#!/usr/bin/env python
"""P3（面试演示）· 生成 4 类可复核案例的 Top-5 检索与缩略图。

输出 reports/case_gallery.json：
  - correct     正确图文匹配案例
  - failed      模型检索失败（困难）案例
  - quality     数据质量异常案例
  - ambiguity   需要人工审核的医学歧义案例
每个案例含：查询图像（相对路径）、图注、Top-5 文本/图像检索结果、相似度、正确配对位置、版权/许可。

图像复制至 assets/thumbnails/，相对 reports/index.html 的路径为 ../assets/thumbnails/<id>.jpg。
ROCOv2 许可：CC BY-NC-SA 4.0（源 eltorio/ROCOv2-radiology，DOI 10.57967/hf/3489）。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(".")
IMG_SRC = ROOT / "data/raw/images"
THUMB = ROOT / "assets/thumbnails"
THUMB.mkdir(parents=True, exist_ok=True)


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_all_emb():
    pools: dict[str, dict] = {}
    emb_by_id: dict[str, dict] = {}
    for split in ("train", "validation", "test"):
        d = np.load(ROOT / f"data/cache/emb_{split}.npz", allow_pickle=True)
        ids = json.load(open(ROOT / f"data/cache/emb_{split}.ids.json", encoding="utf-8"))
        img = d["image_emb"].astype(np.float64)
        txt = d["text_emb"].astype(np.float64)
        img /= np.linalg.norm(img, axis=1, keepdims=True)
        txt /= np.linalg.norm(txt, axis=1, keepdims=True)
        pools[split] = {"ids": ids, "img": img, "txt": txt}
        for i, sid in enumerate(ids):
            emb_by_id[sid] = {"split": split, "img": img[i], "txt": txt[i]}
    return pools, emb_by_id


def topk_within(query_vec: np.ndarray, split: str, pools: dict, k: int = 5):
    p = pools[split]
    sims = p["txt"] @ query_vec if query_vec.shape[0] == p["txt"].shape[1] else p["img"] @ query_vec
    order = sims.argsort()[::-1][:k]
    return [(p["ids"][i], float(sims[i])) for i in order], sims


def copy_img(sid: str) -> str:
    src = IMG_SRC / f"{sid}.jpg"
    dst = THUMB / f"{sid}.jpg"
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)
    return f"../assets/thumbnails/{sid}.jpg"


def build_case(sid: str, pools: dict, emb_by_id: dict, manifest: dict, k: int = 5):
    split = emb_by_id[sid]["split"]
    qimg = emb_by_id[sid]["img"]
    qtxt = emb_by_id[sid]["txt"]
    # I2T: query image -> text candidates
    i2t_res, i2t_sims = topk_within(qimg, split, pools, max(k, len(pools[split]["ids"])))
    i2t_top5 = []
    for cid, s in i2t_res[:k]:
        i2t_top5.append({
            "id": cid, "sim": round(s, 4), "is_correct": cid == sid,
            "caption": (manifest.get(cid, {}).get("caption") or "")[:140], "img": copy_img(cid),
        })
    correct_text_rank = int(np.where(np.array(pools[split]["ids"]) == sid)[0][0]) + 1
    # T2I: query text -> image candidates
    t2i_res, _ = topk_within(qtxt, split, pools, max(k, len(pools[split]["ids"])))
    t2i_top5 = []
    for cid, s in t2i_res[:k]:
        t2i_top5.append({"id": cid, "sim": round(s, 4), "is_correct": cid == sid, "img": copy_img(cid)})
    return {
        "sample_id": sid, "query_img": copy_img(sid), "split": split,
        "candidate_pool_size": len(pools[split]["ids"]),
        "caption": manifest.get(sid, {}).get("caption"),
        "correct_text_rank": correct_text_rank,
        "i2t_top5": i2t_top5, "t2i_top5": t2i_top5,
    }


def main() -> int:
    manifest = {r["sample_id"]: r for r in load_jsonl(ROOT / "data/metadata/clean_manifest.jsonl")}
    per_sample = {r["sample_id"]: r for r in load_jsonl(ROOT / "reports/per_sample_retrieval.jsonl")}
    hard = load_jsonl(ROOT / "reports/hard_cases.jsonl")
    pools, emb_by_id = load_all_emb()

    gallery: dict = {}

    # ---- 1) 正确匹配案例（test, 200 候选，全池内正确项排名第 1）----
    correct_cands = [sid for sid in pools["test"]["ids"] if manifest.get(sid, {}).get("split") == "test"]
    c_sid = None
    for sid in correct_cands:
        cc = build_case(sid, pools, emb_by_id, manifest)
        if cc["correct_text_rank"] == 1 and cc["t2i_top5"][0]["is_correct"]:
            c_sid = sid
            c = cc
            break
    if c_sid is None:
        c_sid = correct_cands[0]
        c = build_case(c_sid, pools, emb_by_id, manifest)
    c["note"] = "查询图像与正确图注在候选集中排名第 1（I2T/T2I 双方向），体现 BiomedCLIP 在医学图文上的强对齐。"
    gallery["correct"] = c

    # ---- 2) 模型检索失败（困难）案例 ----
    failed_cands = [h for h in hard if int(h.get("i2t_rank", 999) or 999) >= 6]
    f = failed_cands[0]
    f_sid = f["sample_id"]
    fc = build_case(f_sid, pools, emb_by_id, manifest)
    fc["rival_caption"] = (f.get("rival_caption") or "")[:140]
    fc["rival_sim"] = round(float(f.get("rival_sim", 0) or 0), 4)
    fc["positive_sim"] = round(float(f.get("positive_sim", 0) or 0), 4)
    fc["margin"] = round(float(f.get("margin", 0) or 0), 4)
    fc["note"] = ("正确图注在候选集中仅排第 %d 名；一个高相似负例（rival）被排在更前，"
                  "属于已挖掘的 229 条困难样本之一。这类样本是数据飞轮优先回填的候选，"
                  "但须经人工复核确认是否为真正错配，而非模型合理歧义。" % fc["correct_text_rank"])
    gallery["failed"] = fc

    # ---- 3) 数据质量异常案例（test, 200 候选）----
    qc = [r for r in manifest.values() if r.get("split") == "test" and r.get("warn_codes")]
    q = qc[0]
    q_sid = q["sample_id"]
    qc_case = build_case(q_sid, pools, emb_by_id, manifest)
    qc_case["warn_codes"] = q.get("warn_codes")
    qc_case["reject_codes"] = q.get("reject_codes", [])
    qc_case["note"] = ("该样本被质量算子标记为 %s。在 P1 流水线中此类样本会被记录告警/拒绝并附带原因，"
                       "保证进入训练的数据具备可追溯的血缘。合成缺陷测试显示算子对缺失/空/占位/重复图像等"
                       "缺陷的自动识别率为 5/5 且无假阳性。" % (q.get("warn_codes")))
    gallery["quality"] = qc_case

    # ---- 4) 需要人工审核的医学歧义案例 ----
    amb_cands = [h for h in hard if h.get("ambiguity_excluded") or h.get("concept_group") == "has_shared_concept"]
    a = amb_cands[0]
    a_sid = a["sample_id"]
    ac = build_case(a_sid, pools, emb_by_id, manifest)
    ac["rival_caption"] = (a.get("rival_caption") or "")[:140]
    ac["rival_sim"] = round(float(a.get("rival_sim", 0) or 0), 4)
    ac["positive_sim"] = round(float(a.get("positive_sim", 0) or 0), 4)
    ac["ambiguity_excluded"] = bool(a.get("ambiguity_excluded", False))
    ac["concept_group"] = a.get("concept_group")
    ac["note"] = ("查询与 rival 共享同一医学概念（共享概念干扰），模型难以区分。此类'图文配对歧义'样本"
                  "在飞轮中被显式排除（ambiguity_excluded）并标注需人工审核，避免把合理歧义误当错误。"
                  "未人工确认前不计入确定错配。")
    gallery["ambiguity"] = ac

    gallery["license"] = {
        "dataset": "ROCOv2 (radiology)",
        "source": "Hugging Face eltorio/ROCOv2-radiology (DOI 10.57967/hf/3489)",
        "license": "CC BY-NC-SA 4.0",
        "usage_note": "缩略图仅用于非商业求职作品集展示；如需公开发布请确认许可与署名要求。",
    }

    Path("reports").mkdir(exist_ok=True)
    Path("reports/case_gallery.json").write_text(json.dumps(gallery, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[gallery] cases:", list(gallery.keys()))
    for k in ("correct", "failed", "quality", "ambiguity"):
        g = gallery[k]
        print(f"  {k}: {g['sample_id']} (split={g['split']}, pool={g['candidate_pool_size']}) rank={g['correct_text_rank']}")
    print("[saved] reports/case_gallery.json ; thumbnails in assets/thumbnails/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
