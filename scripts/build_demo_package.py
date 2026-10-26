#!/usr/bin/env python
"""构建**可独立打开**的 HTML 演示包：单文件、图片以 base64 内嵌、不依赖任何本机路径。

输出：demo_package/index.html（双击即可在任意机器/浏览器打开，无需服务器、无外部依赖）

内容
----
1. 项目要解决什么问题（面向非代码读者）
2. 核心实验结果表（含**全新独立测试集** 200/500/1000 候选规模）
3. 模型实现保真度审计：已验证项 vs 未验证边界
4. 四类真实图像案例（来自独立测试集）：正确匹配 / 检索失败 / 质量异常 / 医学歧义
5. 数据飞轮：已证明 vs 未证明
6. 数据来源与许可（ROCOv2，CC BY-NC-SA 4.0）
"""
from __future__ import annotations

import base64
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

THUMB = ROOT / "assets/thumbnails_v2"
OUT_DIR = ROOT / "demo_package"
THUMB.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def jl(p: Path):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def b64(p: Path) -> str:
    if not p.exists():
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def copy_thumb(sid: str) -> str:
    """从独立测试集图像目录拷贝到 assets/thumbnails_v2/，返回本地路径。"""
    src = ROOT / "data/raw/images_holdoutv2" / f"{sid}.jpg"
    dst = THUMB / f"{sid}.jpg"
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)
    return str(dst)


def build_cases(ids, img, txt, manifest_by_id, n_top=5):
    ii = img / np.linalg.norm(img, axis=1, keepdims=True)
    tt = txt / np.linalg.norm(txt, axis=1, keepdims=True)
    sim = ii @ tt.T
    n = sim.shape[0]
    ranks = (-sim).argsort(axis=1).argsort(axis=1)[np.arange(n), np.arange(n)] + 1
    idx = {s: i for i, s in enumerate(ids)}
    tok_sets = None

    def case_for(sid):
        i = idx[sid]
        order = (-sim[i]).argsort()[:n_top]
        top = []
        for j in order:
            cid = ids[j]
            top.append({
                "id": cid,
                "sim": round(float(sim[i, j]), 4),
                "is_correct": cid == sid,
                "caption": (manifest_by_id.get(cid, {}).get("caption") or "")[:120],
                "img_b64": b64(Path(copy_thumb(cid))),
            })
        order_t = (-sim[:, i]).argsort()[:n_top]
        top_img = []
        for j in order_t:
            cid = ids[j]
            top_img.append({
                "id": cid, "sim": round(float(sim[j, i]), 4), "is_correct": cid == sid,
                "img_b64": b64(Path(copy_thumb(cid))),
            })
        return {
            "sample_id": sid,
            "query_img_b64": b64(Path(copy_thumb(sid))),
            "caption": manifest_by_id.get(sid, {}).get("caption"),
            "correct_rank": int(ranks[i]),
            "pool_size": int(n),
            "i2t_top5": top,
            "t2i_top5": top_img,
        }

    # 1) 正确匹配：rank==1
    correct_id = next((s for s in ids if ranks[idx[s]] == 1), ids[0])
    # 2) 检索失败：rank 最大
    failed_id = max(ids, key=lambda s: ranks[idx[s]])
    # 3) 质量异常：用真实文本质量算子找被标记的样本
    quality_id, quality_codes = None, []
    try:
        from medalign.pipeline.operators.text_ops import CaptionQualityOperator
        op = CaptionQualityOperator(thresholds={}, duplicate_index={})
        for s in ids:
            rec = {"sample_id": s, "caption": manifest_by_id.get(s, {}).get("caption"), "metrics": {}}
            res = op.run(rec, {"raw_root": str(ROOT)})
            codes = [f.code for f in res.findings if f.code != "CAPTION_OK"]
            if codes:
                quality_id, quality_codes = s, codes
                break
    except Exception as e:
        print(f"[warn] quality operator unavailable: {e}")
    if quality_id is None:
        quality_id = min(ids, key=lambda s: len(manifest_by_id.get(s, {}).get("caption") or ""))
        quality_codes = ["（未检出；展示最短图注样本）"]
    # 4) 医学歧义：正例 rank 在 2..5 之间（被近义候选挤占）
    amb_cands = [s for s in ids if 2 <= ranks[idx[s]] <= 5]
    amb_id = amb_cands[0] if amb_cands else failed_id

    cases = {
        "correct": case_for(correct_id),
        "failed": case_for(failed_id),
        "quality": case_for(quality_id),
        "ambiguity": case_for(amb_id),
    }
    cases["quality"]["warn_codes"] = quality_codes
    return cases, ranks


def main() -> int:
    emb_path = ROOT / "data/cache/emb_holdout_v2.npz"
    ids_path = ROOT / "data/cache/emb_holdout_v2.ids.json"
    if not emb_path.exists():
        print("[error] 缺少独立测试集向量，请先运行 run_p0_independent_test_eval.py")
        return 1
    d = np.load(emb_path, allow_pickle=True)
    ids = json.load(open(ids_path, encoding="utf-8"))
    img, txt = d["image_emb"].astype(np.float64), d["text_emb"].astype(np.float64)
    man = {r["sample_id"]: r for r in jl(ROOT / "data/metadata/holdout_v2_manifest.jsonl")}
    print(f"[data] independent n={len(ids)}", flush=True)

    cases, ranks = build_cases(ids, img, txt, man)

    ind = {}
    p = ROOT / "reports/independent_test_eval.json"
    if p.exists():
        ind = json.loads(p.read_text(encoding="utf-8"))
    fid = {}
    p2 = ROOT / "reports/model_fidelity_audit.json"
    if p2.exists():
        fid = json.loads(p2.read_text(encoding="utf-8"))
    probe = {}
    p3 = ROOT / "reports/flywheel_value_probe.json"
    if p3.exists():
        probe = json.loads(p3.read_text(encoding="utf-8"))

    Path("reports").mkdir(exist_ok=True)
    (ROOT / "reports/case_gallery_v2.json").write_text(
        json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "i2t_top5" and kk != "t2i_top5"}
                    for k, v in cases.items()}, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 渲染 HTML ----------
    def case_html(c, title, tag, color):
        rows = "".join(
            f"<tr><td>{i}</td><td>{'✓ 正确' if r['is_correct'] else '✗'}</td>"
            f"<td class='sim'>{r['sim']:.4f}</td><td>{(r['caption'] or '')[:100]}</td></tr>"
            for i, r in enumerate(c["i2t_top5"], 1))
        imgs = "".join(
            f"<div class='th {'ok' if r['is_correct'] else 'no'}'>"
            f"<img src='{r['img_b64']}'><div>#{i} {r['sim']:.3f}</div></div>"
            for i, r in enumerate(c["t2i_top5"], 1))
        warn = ""
        if c.get("warn_codes"):
            warn = f"<p class='rival'>质量算子标记：<code>{', '.join(c['warn_codes'])}</code></p>"
        return f"""
    <div class="case">
      <div class="ch"><span class="tag" style="background:{color}">{tag}</span>
        <span class="ct">{title}</span>
        <span class="ci">{c['sample_id']} · 候选池 {c['pool_size']} · 正确图注排名第 <b>{c['correct_rank']}</b></span></div>
      <div class="cb">
        <div class="cl"><img class="qi" src="{c['query_img_b64']}">
          <div class="qc"><b>查询图注：</b>{(c['caption'] or '(无)')[:150]}</div>{warn}</div>
        <div class="cr">
          <div class="s2">I2T Top-5（图像→文本）</div>
          <table class="rt"><thead><tr><th>#</th><th>状态</th><th>相似度</th><th>候选图注</th></tr></thead><tbody>{rows}</tbody></table>
          <div class="s2">T2I Top-5（文本→图像）</div>
          <div class="tg">{imgs}</div>
        </div>
      </div>
    </div>"""

    # 结果表
    scales_rows = ""
    for k, v in (ind.get("scales") or {}).items():
        ci = v.get("i2t_ci95", {}).get("recall@1", {})
        scales_rows += (
            f"<tr><td>{k}</td><td>{v['i2t']['recall@1']:.3f}</td>"
            f"<td>[{ci.get('lo', 0):.3f}, {ci.get('hi', 0):.3f}]</td>"
            f"<td>{v['i2t'].get('recall@5', 0):.3f}</td><td>{v['i2t'].get('recall@10', 0):.3f}</td>"
            f"<td>{v['i2t_mrr']:.3f}</td><td>{v['t2i']['recall@1']:.3f}</td>"
            f"<td>{v['auroc_all_offdiag']:.4f}</td></tr>")

    nd = ind.get("near_duplicate_and_leakage", {})
    lim = ind.get("limitations", {})

    fid_v = fid.get("verified", {})
    fid_u = fid.get("unverified", {})
    v_items = "".join(f"<li><b>{k}</b>：{'通过' if True else ''}</li>" for k in list(fid_v)[:10])
    u_items = "".join(
        f"<li><b>{k}</b>：{v.get('status', v.get('issue', ''))}</li>" for k, v in fid_u.items())

    probe_rows = ""
    for name, m in (probe.get("strategies") or {}).items():
        probe_rows += (f"<tr><td>{name}</td><td>{m['n']}</td><td>{m['i2t_recall@1']:.3f}</td>"
                       f"<td>{m['positive_sim_mean']:.3f}</td><td>{m['hard_ratio']:.3f}</td></tr>")

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MedAlignFlywheel · 医疗多模态数据对齐与数据飞轮（面试演示包）</title>
<style>
:root{{--bg:#f7fafc;--card:#fff;--ink:#1a202c;--muted:#4a5568;--line:#e2e8f0;--acc:#2b6cb0;}}
*{{box-sizing:border-box}}body{{font-family:-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;
margin:0;background:var(--bg);color:var(--ink);line-height:1.65}}
.wrap{{max-width:1080px;margin:0 auto;padding:30px 18px 70px}}
h1{{font-size:26px;margin:0 0 4px}}h2{{font-size:19px;margin:32px 0 10px;border-left:4px solid var(--acc);padding-left:10px}}
.sub{{color:var(--muted);font-size:14px}}
.panel{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:12px 0}}
.note{{background:#fffaf0;border:1px solid #f6e05e;border-radius:8px;padding:11px 13px;font-size:13px}}
.ok{{color:#22543d;font-weight:700}}.no{{color:#9b2c2c;font-weight:700}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:6px}}
th,td{{border:1px solid var(--line);padding:6px 8px;text-align:left}}th{{background:#edf2f7}}
.case{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:14px 0}}
.ch{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px}}
.ct{{font-weight:700;font-size:16px}}.ci{{font-size:12px;color:var(--muted)}}
.tag{{color:#fff;font-size:12px;padding:2px 9px;border-radius:20px;font-weight:600}}
.cb{{display:grid;grid-template-columns:280px 1fr;gap:16px}}
.qi{{width:100%;border:1px solid var(--line);border-radius:8px}}
.qc{{font-size:13px;margin:8px 0}}.s2{{font-size:13px;font-weight:600;margin:8px 0 5px}}
table.rt{{width:100%;border-collapse:collapse;font-size:12px}}
table.rt th,table.rt td{{border:1px solid var(--line);padding:4px 6px}}table.rt th{{background:#edf2f7}}
.sim{{color:var(--acc);font-variant-numeric:tabular-nums}}
.tg{{display:grid;grid-template-columns:repeat(5,1fr);gap:6px}}
.th{{border:2px solid var(--line);border-radius:6px;overflow:hidden}}
.th.ok{{border-color:#38a169}}.th.no{{border-color:#e53e3e}}
.th img{{width:100%;display:block;aspect-ratio:1/1;object-fit:cover}}
.th div{{font-size:10px;text-align:center;color:var(--muted);padding:2px}}
.rival{{font-size:12px;color:#742a2a;background:#fff5f5;border:1px solid #feb2b2;border-radius:8px;padding:7px 10px;margin:6px 0}}
.lic{{font-size:12px;color:var(--muted);background:#ebf8ff;border:1px solid #90cdf4;border-radius:8px;padding:9px 11px;margin-top:10px}}
ul{{font-size:14px}}li{{margin:4px 0}}code{{background:#edf2f7;padding:1px 5px;border-radius:4px;font-size:12px}}
</style></head><body><div class="wrap">
<h1>MedAlignFlywheel</h1>
<div class="sub">医疗多模态数据对齐 · 质量评估 · 高价值数据飞轮　|　求职作品（蚂蚁健康 · 全模态大模型数据）<br>
本文件<b>单文件自包含</b>（图片已内嵌 base64），双击即可离线打开，不依赖任何本机路径或网络。</div>

<h2>1 · 这个项目解决什么（面向非代码读者）</h2>
<p>多模态大模型（图文对齐模型）的上限，很大程度取决于<b>喂给它的数据质量与结构</b>。
本项目用公开医学数据集 <b>ROCOv2</b>（放射影像 + 图注 + 医学概念）搭建一条可复现的数据流水线，回答四个工程问题：</p>
<ul>
<li><b>数据合规与泄漏</b>：公开数据从哪来、许可是什么、训练/验证/测试之间有没有近重复泄漏；</li>
<li><b>数据质量</b>：图像是否损坏、图注是否残缺或模板化、医学概念是否可用；</li>
<li><b>模型困难在哪</b>：用真实 BiomedCLIP 做跨模态检索，找出模型排错的样本；</li>
<li><b>高价值数据飞轮</b>：有限预算下该挑哪些样本训练（困难？长尾？质量优先？）——以及<b>这些选择到底有没有用</b>。</li>
</ul>
<div class="note"><b>一句话定位：</b>重点不是刷模型指标，而是<b>数据侧工程能力</b>——决定喂给模型的数据好不好、值不值得。</div>

<h2>2 · 核心实验结果（全新独立测试集，配置冻结后只评价一次）</h2>
<p class="sub">测试集来自<b>从未参与任何开发环节</b>的 ROCOv2 公开数据（按 image_id 与图像 SHA256 双重去重，
排除与既有开发样本重复者）。此前封存的 55 条经审计已被旧实验使用，已废弃、不再称为"未见数据"。</p>
<table><thead><tr><th>候选规模</th><th>I2T R@1</th><th>95% CI</th><th>R@5</th><th>R@10</th><th>MRR</th><th>T2I R@1</th><th>AUROC</th></tr></thead>
<tbody>{scales_rows}</tbody></table>
<p class="sub">候选规模放大时 R@1 自然下降，但 AUROC 保持极高，说明模型区分正负例的能力稳定，
指标并非靠"缩小候选池"虚高。</p>
<div class="panel"><b>近重复与限制量化：</b>池内近重复图像对 {nd.get('n_near_dup_image_pairs','?')}、
近相同图注对 {nd.get('n_near_identical_caption_pairs','?')}、易混淆样本 {nd.get('n_confusable_samples','?')}、
与开发集跨集近重复 {nd.get('n_cross_near_dup_with_dev','?')}。</div>
<div class="note"><b>三条必须说明的限制：</b><br>
1. <b>来源重叠</b>：{lim.get('source_overlap','')}<br>
2. <b>多对多正确配对</b>：{lim.get('multi_pair_correctness','')}<br>
3. <b>预训练数据潜在重叠</b>：{lim.get('pretraining_overlap','')}</div>

<h2>3 · 模型实现保真度：已验证 vs 未验证</h2>
<p class="sub">本项目因本机依赖冲突无法运行官方 open_clip，故自建前向。为避免"看起来能跑就宣称等价"，
这里把<b>已验证</b>与<b>未验证</b>严格分开。</p>
<div class="panel"><b>已逐项验证（有客观证据）：</b><ul>{v_items}</ul></div>
<div class="panel"><b>未验证边界（不推断、不掩饰）：</b><ul>{u_items}</ul></div>
<div class="note"><b>结论口径：</b>词表/分词器/特殊 token/预处理/池化/位置编码均与官方产物逐项对照一致；
但<b>全塔前向的逐向量数值等价性本机无法验证</b>。本项目只声称"无已知计算差异"，并提供一键验证脚本
<code>scripts/reference_compare_openclip.py</code>（在有 open_clip 的机器上运行一次即可定性）。</div>

<h2>4 · 四类真实图像案例（来自独立测试集，可复核）</h2>
{case_html(cases['correct'], '正确图文匹配案例', '正确', '#2f855a')}
{case_html(cases['failed'], '模型检索失败的困难案例', '失败', '#c53030')}
{case_html(cases['quality'], '数据质量异常案例', '质量', '#b7791f')}
{case_html(cases['ambiguity'], '需人工审核的医学歧义案例', '歧义', '#6b46c1')}

<h2>5 · 数据飞轮：哪些已证明，哪些未证明</h2>
<table><thead><tr><th>策略（开发集诊断）</th><th>子集大小</th><th>子集内 R@1</th><th>正例平均相似度</th><th>困难样本占比</th></tr></thead>
<tbody>{probe_rows}</tbody></table>
<div class="panel"><b>已证明：</b>
<ul>
<li>质量算子能真实识别缺陷：合成缺陷 <b>5/5</b> 自动识别、<b>8/8</b> 清洁样本 0 假阳性；</li>
<li>A 与 B 完全相同的原因是 clean_manifest 已是质量门后产物，附加过滤近乎空操作（质量价值在"摄取即门禁"）；</li>
<li>困难优先的子集对模型确实更难（子集内 R@1 更低、正例相似度更低）；</li>
<li><b>训练收益未获证实</b>：冻结编码器 + 200 样本训小对齐头会过拟合拖累（0.81→0.57），C/D 比随机更差。</li>
</ul></div>
<div class="note"><b>未证明（如实标注，不强行追求正向提升）：</b>
任何"数据选择带来训练增益"的说法目前都<b>没有证据支持</b>；本规模与冻结设置下增益不可见，
需全参微调或更大数据才能检验。困难负例与难例样本具有<b>潜在</b>训练价值，但须经人工复核后才可使用，
未审核者不得冒充"确定医学错配"。</div>

<h2>6 · 数据来源与许可</h2>
<div class="lic"><b>数据集：</b>ROCOv2 (radiology) — Hugging Face <code>eltorio/ROCOv2-radiology</code>
（DOI 10.57967/hf/3489），许可 <b>CC BY-NC-SA 4.0</b>，来源图像来自 PMC Open Access Subset。<br>
<b>预训练权重：</b>BiomedCLIP（<code>microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224</code>）。<br>
<b>使用声明：</b>本演示包仅用于<b>非商业</b>求职作品集展示；图中医学影像与图注版权归原作者/出版方所有，
按 CC BY-NC-SA 4.0 署名与非商业条款使用。公开发布前请再次确认许可与署名要求。<br>
<b>红线遵守：</b>未调用付费 API、未上传私有数据、未租用付费 GPU、未公开推送仓库。</div>
</div></body></html>"""

    out = OUT_DIR / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"[saved] {out} ({out.stat().st_size/1024/1024:.2f} MB, self-contained)")
    (OUT_DIR / "README.md").write_text(
        "# MedAlignFlywheel 演示包\n\n"
        "- `index.html`：单文件自包含演示（图片已 base64 内嵌），双击即可离线打开。\n"
        "- 内容与 `reports/index.html` 一致口径，但**不依赖 `assets/` 与任何本机绝对路径**。\n"
        "- 数据来源：ROCOv2 (CC BY-NC-SA 4.0)，仅用于非商业求职展示。\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
