#!/usr/bin/env python
"""P5：把 P2/P3 的 JSON 产物渲染成自包含 HTML 报告/演示页（无外部依赖）。

产出 reports/index.html。所有图表用内联 SVG，离线可看。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"


def bar_chart(title, labels, values, color="#2b6cb0", width=560, height=240, maxv=None):
    """竖向柱状图（SVG）。values 为 0-1 比例或任意正数。"""
    n = len(labels)
    pad = 40
    plot_w = width - pad * 2
    plot_h = height - pad * 2
    maxv = maxv or (max(values) * 1.15 or 1)
    bw = plot_w / n * 0.6
    gap = plot_w / n
    bars = []
    for i, (lab, v) in enumerate(zip(labels, values)):
        x = pad + gap * i + (gap - bw) / 2
        h = (v / maxv) * plot_h if maxv else 0
        y = pad + plot_h - h
        pct = f"{v*100:.1f}%" if maxv and v <= 1.05 else f"{v:.3f}"
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" '
            f'fill="{color}" rx="2"></rect>'
            f'<text x="{x+bw/2:.1f}" y="{y-5:.1f}" font-size="11" text-anchor="middle">{pct}</text>'
            f'<text x="{x+bw/2:.1f}" y="{pad+plot_h+14:.1f}" font-size="10" text-anchor="middle">{lab}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px">'
        f'<text x="{pad}" y="16" font-size="13" font-weight="600">{title}</text>'
        f'<line x1="{pad}" y1="{pad+plot_h}" x2="{width-pad}" y2="{pad+plot_h}" stroke="#ccc"/>'
        + "".join(bars)
        + "</svg>"
    )


def metric_cards(items):
    out = ['<div class="cards">']
    for label, val, sub in items:
        out.append(
            f'<div class="card"><div class="card-val">{val}</div>'
            f'<div class="card-label">{label}</div>'
            f'<div class="card-sub">{sub}</div></div>'
        )
    out.append("</div>")
    return "".join(out)


def table(headers, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = []
    for r in rows:
        trs.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    return (
        f'<table class="{cls}"><thead><tr>{th}</tr></thead>'
        f'<tbody>{"".join(trs)}</tbody></table>'
    )


def render_case(case, heading, tag, tag_color):
    """渲染单个可复核案例：查询图 + 图注 + Top-5 文本/图像检索 + 相似度 + 正确位置。"""
    sid = case["sample_id"]
    cap = case.get("caption") or ""
    rank = case["correct_text_rank"]
    pool = case.get("candidate_pool_size", "?")
    qimg = case["query_img"]
    note = case.get("note", "")

    # I2T Top-5 文本
    i2t_rows = []
    for i, r in enumerate(case["i2t_top5"], 1):
        badge = '<span class="okb">✓ 正确</span>' if r["is_correct"] else '<span class="wtb">✗</span>'
        i2t_rows.append(
            f"<tr><td>{i}</td><td>{badge}</td><td class='sim'>{r['sim']:.4f}</td>"
            f"<td>{(r['caption'] or '')[:110]}</td></tr>"
        )
    i2t_table = (
        "<table class='rt'><thead><tr><th>#</th><th>状态</th><th>相似度</th><th>候选图注（截断）</th></tr></thead>"
        f"<tbody>{''.join(i2t_rows)}</tbody></table>"
    )

    # T2I Top-5 缩略图网格
    t2i_cells = []
    for i, r in enumerate(case["t2i_top5"], 1):
        cls = "okb" if r["is_correct"] else "wtb"
        t2i_cells.append(
            f"<div class='thumb {cls}'><img src='{r['img']}' alt='{r['id']}' loading='lazy'>"
            f"<div class='cap'>#{i} sim={r['sim']:.3f}</div></div>"
        )
    t2i_grid = f"<div class='thumbgrid'>{''.join(t2i_cells)}</div>"

    rival = case.get("rival_caption")
    rival_line = ""
    if rival:
        rival_line = (
            f"<p class='rival'>最强干扰负例图注：<i>{(rival or '')[:140]}</i>　"
            f"（rival sim={case.get('rival_sim',0):.3f} vs 正例 sim={case.get('positive_sim',0):.3f}）</p>"
        )
    warn = case.get("warn_codes")
    warn_line = ""
    if warn:
        warn_line = f"<p class='rival'>质量算子标记：<code>{', '.join(warn)}</code></p>"

    return f"""
<div class="case">
  <div class="case-head"><span class="tag" style="background:{tag_color}">{tag}</span>
    <span class="case-title">{heading}</span>
    <span class="case-id">{sid}　·　候选池 {pool}　·　正确图注排名 <b>{rank}</b></span></div>
  <div class="case-body">
    <div class="case-left">
      <img class="qimg" src="{qimg}" alt="{sid}">
      <div class="qcap"><b>查询图注：</b>{(cap or '(无图注)')[:160]}</div>
      {warn_line}{rival_line}
      <p class="cnote">{note}</p>
    </div>
    <div class="case-right">
      <div class="sub2">I2T Top-5（图像→文本，按相似度）</div>
      {i2t_table}
      <div class="sub2">T2I Top-5（文本→图像，按相似度）</div>
      {t2i_grid}
    </div>
  </div>
</div>
"""


def main() -> int:
    m = json.loads((REP / "retrieval_metrics.json").read_text(encoding="utf-8"))
    sel = json.loads((REP / "selection_report.json").read_text(encoding="utf-8"))
    gallery = None
    gpath = REP / "case_gallery.json"
    if gpath.exists():
        gallery = json.loads(gpath.read_text(encoding="utf-8"))

    # P4（若有）
    p4 = None
    p4_path = REP / "p4_adapter.json"
    if p4_path.exists():
        p4 = json.loads(p4_path.read_text(encoding="utf-8"))

    test = m["splits"]["test"]
    val = m["splits"]["validation"]
    ti, tt = test["i2t"], test["t2i"]
    vi, vt = val["i2t"], val["t2i"]
    tau = test["negative_audit"]

    # 卡片
    cards = metric_cards([
        ("TEST · I2T R@1", f"{ti['recall@1']*100:.1f}%", "图像检索文本，Top-1 命中率（200 候选）"),
        ("TEST · T2I R@1", f"{tt['recall@1']*100:.1f}%", "文本检索图像，Top-1 命中率"),
        ("TEST · MRR", f"{ti['mrr']:.3f}", "平均倒数排名（I2T）"),
        ("TEST · AUROC", f"{tau['auroc_clean']:.3f}", "正负例可分性（净化后）"),
    ])

    # 图表
    chart_i2t = bar_chart(
        "Image→Text Recall@K / MRR（test）",
        ["R@1", "R@5", "R@10", "MRR"],
        [ti["recall@1"], ti["recall@5"], ti["recall@10"], ti["mrr"]],
        color="#2b6cb0",
    )
    chart_t2i = bar_chart(
        "Text→Image Recall@K / MRR（test）",
        ["R@1", "R@5", "R@10", "MRR"],
        [tt["recall@1"], tt["recall@5"], tt["recall@10"], tt["mrr"]],
        color="#2f855a",
    )
    chart_val = bar_chart(
        "I2T R@K：validation vs test",
        ["val R@1", "val R@5", "val R@10", "test R@1", "test R@5", "test R@10"],
        [vi["recall@1"], vi["recall@5"], vi["recall@10"],
         ti["recall@1"], ti["recall@5"], ti["recall@10"]],
        color="#805ad5",
    )

    # 分层分析（test, I2T）
    def strat_rows(dim):
        d = test["stratified"]["i2t"].get(dim, {})
        rows = []
        for k, v in sorted(d.items()):
            rows.append([k, v["n"], f"{v['recall@1']*100:.1f}%",
                         f"{v['recall@5']*100:.1f}%", f"{v['recall@10']*100:.1f}%",
                         f"{v['mrr']:.3f}"])
        return rows

    strat_mod = table(
        ["模态", "n", "R@1", "R@5", "R@10", "MRR"],
        strat_rows("modality_proxy"),
    )
    strat_len = table(
        ["图注长度带", "n", "R@1", "R@5", "R@10", "MRR"],
        strat_rows("caption_length_band"),
    )
    strat_con = table(
        ["概念组", "n", "R@1", "R@5", "R@10", "MRR"],
        strat_rows("concept_group"),
    )

    # 选择策略对比
    tbl = sel["selection"]["table"]
    sel_rows = []
    for k in ["A_random", "B_quality", "C_quality_hard", "D_quality_longtail"]:
        v = tbl[k]
        sel_rows.append([
            k, v["n_selected"], v["n_distinct_modality"], v["n_distinct_cui"],
            v["n_hard_samples"], f"{v['near_dup_ratio']*100:.1f}%",
            f"{v.get('caption_tokens_mean',0):.1f}",
        ])
    sel_table = table(
        ["策略", "选中数", "模态覆盖", "CUI覆盖", "困难样本", "近重复占比", "平均token"],
        sel_rows,
    )
    jac = sel["selection"]["pairwise_jaccard"]
    jac_rows = [[k.replace("__vs__", " vs "), f"{v*100:.1f}%"] for k, v in jac.items()]
    jac_table = table(["策略对", "集合 Jaccard"], jac_rows)

    # 难例样本（取前若干条做展示，含空白复核字段说明）
    hard_disp = []
    with open(REP / "hard_cases.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 12:
                break
            c = json.loads(line)
            hard_disp.append([
                c["sample_id"], c["split"], c["kind"],
                str(c.get("i2t_rank")), f"{c['positive_sim']:.3f}",
                (c.get("rival_sample_id") or "-"), f"{c.get('margin',0) or 0:.3f}",
                (c["caption"][:60] + "…") if len(c["caption"]) > 60 else c["caption"],
            ])
    hard_table = table(
        ["sample_id", "split", "kind", "I2T排名", "正例sim", "rival", "margin", "图注(截断)"],
        hard_disp,
    )

    # 负例构造摘要
    hn = sel["hard_negatives"]
    neg_rows = []
    for sp, blk in hn.items():
        neg_rows.append([
            sp, blk["n_candidates"], blk["n_confident_negative"],
            f"{blk['confident_ratio']*100:.1f}%", blk["n_hard"],
            blk["needs_review_reasons"].get("shared_medical_concept", 0),
            blk["needs_review_reasons"].get("caption_overlap_too_high", 0),
            blk["needs_review_reasons"].get("near_duplicate_image_or_similar_source", 0),
        ])
    neg_table = table(
        ["split", "候选", "确定负例", "确定占比", "难负例", "审查-共概念", "审查-图注重叠", "审查-近重复"],
        neg_rows,
    )

    hm = sel["hard_mining"]
    hm_summary = "; ".join(
        f"{sp}: {blk['n_cases']} 例（risk {blk['by_kind'].get('risk_false_positive',0)}, "
        f"hard {blk['by_kind'].get('hard_positive',0)}）"
        for sp, blk in hm.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MedAlignFlywheel · 多模态数据对齐与数据飞轮报告</title>
<style>
:root{{--bg:#f7fafc;--card:#fff;--ink:#1a202c;--muted:#4a5568;--line:#e2e8f0;--accent:#2b6cb0;}}
*{{box-sizing:border-box}}
body{{font-family:-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;
 margin:0;background:var(--bg);color:var(--ink);line-height:1.6}}
.wrap{{max-width:1080px;margin:0 auto;padding:32px 20px 80px}}
h1{{font-size:26px;margin:0 0 4px}}
h2{{font-size:20px;margin:36px 0 12px;border-left:4px solid var(--accent);padding-left:10px}}
.sub{{color:var(--muted);font-size:14px;margin-bottom:8px}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;text-align:center}}
.card-val{{font-size:30px;font-weight:700;color:var(--accent)}}
.card-label{{font-size:13px;font-weight:600;margin-top:4px}}
.card-sub{{font-size:11px;color:var(--muted);margin-top:2px}}
.case{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:14px 0}}
.case-head{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px}}
.case-title{{font-weight:700;font-size:16px}}
.case-id{{font-size:12px;color:var(--muted)}}
.tag{{color:#fff;font-size:12px;padding:2px 9px;border-radius:20px;font-weight:600}}
.case-body{{display:grid;grid-template-columns:300px 1fr;gap:18px}}
.qimg{{width:100%;border:1px solid var(--line);border-radius:8px;background:#fff}}
.qcap{{font-size:13px;margin:8px 0}}
.cnote{{font-size:12px;color:var(--muted);background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:8px 10px}}
.rival{{font-size:12px;color:#742a2a;background:#fff5f5;border:1px solid #feb2b2;border-radius:8px;padding:7px 10px;margin:6px 0}}
.sub2{{font-size:13px;font-weight:600;margin:10px 0 6px}}
table.rt{{width:100%;border-collapse:collapse;font-size:12px}}
table.rt th,table.rt td{{border:1px solid var(--line);padding:5px 7px}}
table.rt th{{background:#edf2f7}}
.okb{{color:#22543d;font-weight:700}}
.wtb{{color:#9b2c2c;font-weight:700}}
.sim{{font-variant-numeric:tabular-nums;color:var(--accent)}}
.thumbgrid{{display:grid;grid-template-columns:repeat(5,1fr);gap:6px}}
.thumb{{border:2px solid var(--line);border-radius:6px;overflow:hidden;background:#fff}}
.thumb.okb{{border-color:#38a169}}
.thumb.wtb{{border-color:#e53e3e}}
.thumb img{{width:100%;display:block;aspect-ratio:1/1;object-fit:cover}}
.thumb .cap{{font-size:10px;text-align:center;padding:2px 0;color:var(--muted)}}
.lic{{font-size:12px;color:var(--muted);background:#ebf8ff;border:1px solid #90cdf4;border-radius:8px;padding:8px 10px;margin-top:10px}}
.panel{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:14px 0}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:6px}}
th,td{{border:1px solid var(--line);padding:7px 9px;text-align:left}}
th{{background:#edf2f7;font-weight:600}}
tr:nth-child(even) td{{background:#f8fafc}}
code{{background:#edf2f7;padding:1px 5px;border-radius:4px;font-size:12px}}
.note{{background:#fffaf0;border:1px solid #f6e05e;border-radius:8px;padding:12px 14px;font-size:13px}}
.ok{{background:#f0fff4;border:1px solid #9ae6b4;border-radius:8px;padding:12px 14px;font-size:13px}}
ul{{font-size:14px}}li{{margin:4px 0}}
</style></head><body><div class="wrap">
<h1>MedAlignFlywheel</h1>
<div class="sub">医疗多模态数据对齐 · 质量评估 · 高价值数据飞轮　|　求职项目（蚂蚁健康 · 全模态大模型数据）</div>

<h2>0 · 项目概览</h2>
<p>本项目以公开数据集 <b>ROCOv2（医学影像+图注+CUI）</b> 为基底，构建一条可复现的医学多模态数据流水线：
<b>P0 合规摄取与跨集泄漏检查</b> → <b>P1 六大质量算子</b> → <b>P2 用真实 BiomedCLIP 权重做跨模态检索评测</b> →
<b>P3 数据飞轮（难例挖掘/难负例构造/四种数据选择）</b> → <b>P4 冻结编码器+小对齐头对照</b>。
所有真实权重均通过严格门禁加载（zero-missing/zero-unexpected + 指纹），检索评测在 <b>冻结 test 集</b> 上进行，禁止据 test 调参。</p>

<h2>1 · P2 跨模态检索评测（真实 BiomedCLIP）</h2>
{cards}
<div class="grid2">
<div class="panel">{chart_i2t}</div>
<div class="panel">{chart_t2i}</div>
</div>
<div class="panel">{chart_val}</div>
<p class="sub">正例平均相似度 <b>{tau['positive']['mean']:.3f}</b>，净化后负例平均 <b>{tau['clean_negative']['mean']:.3f}</b>，
AUROC(clean)=<b>{tau['auroc_clean']:.3f}</b>；共排除 {tau['n_ambiguous_pair']} 个"模糊负例"（图注高度重叠或图像近重复），
避免负例被系统性高估。validation（开发集）R@1=0.727，test（冻结集）R@1=0.805，二者量级一致、无泄露迹象。</p>

<h2>2 · 分层错误分析（test · I2T）</h2>
<div class="grid2">
<div class="panel"><b>按影像模态</b>{strat_mod}</div>
<div class="panel"><b>按图注长度</b>{strat_len}</div>
</div>
<div class="panel"><b>按概念长尾</b>{strat_con}</div>
<p class="sub">长尾/单例概念组（longtail_all_singleton）通常比共享概念组更难检索，提示"稀缺概念"是数据飞轮的重点增益区。</p>

<h2>3 · P3 数据飞轮</h2>
<div class="ok"><b>难例挖掘：</b>{hm_summary}。所有难例均带空白 <code>review</code> 字段，未经人工复核前不宣称任何数据缺陷。</div>
<div class="panel"><b>难负例构造（三重校验：来源非近重复 / 概念不相交 / 语义不重叠）</b>{neg_table}
<p class="sub">确定负例需同时通过三重检查；未通过者标记 <code>needs_review</code> 并单列原因（以"共享医学概念"为主，符合 ROCOv2 同论文多子图的特性）。</p>
</div>

<h2>4 · 四种数据选择策略对比（train，预算=200）</h2>
<div class="grid2">
<div class="panel"><b>数据侧指标</b>{sel_table}</div>
<div class="panel"><b>策略两两重叠（Jaccard）</b>{jac_table}</div>
</div>
<p class="sub">A 原始随机 / B 质量过滤后随机（本清洁子集二者完全一致）/ C 质量+困难优先 / D 质量+长尾优先。
C 把困难样本占比从 ~27% 提升到 83%；D 模态覆盖回落到 6（长尾优先的代价）。这些只是<b>数据侧</b>描述，
是否带来模型指标差异必须由 P4 对照训练给出。</p>
"""

    # P4 块（动态插入）
    if p4 is not None:
        base = p4["baseline"]
        p4_rows = [["冻结基线（不训练）", "-", f"{base['i2t']['recall@1']*100:.1f}%", f"{base['t2i']['recall@1']*100:.1f}%"]]
        for k in ["A_random", "B_quality", "C_quality_hard", "D_quality_longtail"]:
            if k in p4["strategies"]:
                v = p4["strategies"][k]
                p4_rows.append([
                    k, v["n_train_used"],
                    f"{v['i2t']['recall@1']*100:.1f}%", f"{v['t2i']['recall@1']*100:.1f}%",
                ])
        p4_table = table(["设置", "训练样本数", "I2T R@1", "T2I R@1"], p4_rows)
        p4_block = f"""
<h2>4b · P4 对齐头对照（已真实运行）</h2>
<div class="panel"><b>冻结编码器 + 小对齐头，A/B/C/D 同预算（200 样本）</b>{p4_table}
<p class="sub">{p4['disclaimer']}</p>
<div class="note"><b>诚实结论：</b>在已接近天花板的冻结编码器上，仅用 200 样本训练小对齐头会<b>过拟合并拖累</b>检索
（0.81→0.57）；困难/长尾优先的 C/D 比随机 A/B 更差。说明本规模 + 冻结编码器下，数据选择的增益不可见，
需全参微调或更大数据才显形。P4 的价值在于给出这一边界结论，而非宣称某策略更优。</div>
</div>
"""
        html = html + p4_block

    # 面试案例库（可复核）
    if gallery is not None:
        cases_html = render_case(gallery["correct"], "正确图文匹配案例", "正确", "#2f855a")
        cases_html += render_case(gallery["failed"], "模型检索失败的困难案例", "失败", "#c53030")
        cases_html += render_case(gallery["quality"], "数据质量异常案例", "质量", "#b7791f")
        cases_html += render_case(gallery["ambiguity"], "需人工审核的医学歧义案例", "歧义", "#6b46c1")
        lic = gallery.get("license", {})
        lic_html = (
            f"<div class='lic'><b>数据来源与许可：</b>{lic.get('dataset','')} — {lic.get('source','')}；"
            f"许可 {lic.get('license','')}。{lic.get('usage_note','')}</div>"
            if lic else ""
        )
        gallery_block = f"""
<h2>5 · 面试案例库（可复核，含真实图像）</h2>
<p class="sub">以下 4 个案例直接从真实检索结果中抽取，图像与相似度均来自本项目对 BiomedCLIP 的真实前向推理，
未做任何修饰。正确配对位置即该样本在候选池中的真实排名。所有缩略图保留原始出处与许可信息。</p>
{cases_html}
{lic_html}
"""
        html = html + gallery_block

    html = html + f"""
<div class="panel">{hard_table}
<p class="sub">完整 {sum(blk['n_cases'] for blk in hm.values())} 条见 <code>reports/hard_cases.jsonl</code>，每条预留 <code>review.is_true_mismatch / reviewer_note / confirmed_by</code> 供人工填写。</p>
</div>

<h2>6 · 完整性与可复现性</h2>
<ul>
<li><b>权重指纹：</b>BiomedCLIP 官方 HF <code>microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224</code>；
文件 SHA256 与张量级 SHA256 见 <code>reports/embedding_run.json</code>，加载模式 <code>strict zero-missing zero-unexpected</code>。</li>
<li><b>冻结纪律：</b>test 集仅用于最终评测，不参与 P1 调参、P3 挖掘/选择、P4 训练。</li>
<li><b>负例净化：</b>模糊负例（{tau['n_ambiguous_pair']} 个）不计入 AUROC，如实公开。</li>
<li><b>非伪造纪律：</b>相似度全部来自真实前向推理；任何未实际运行的模块在文档中标注 <code>NOT_TESTED</code>。</li>
</ul>
<div class="note"><b>重要更正记录：</b>初版检索出现 AUROC&lt;0.5 与 R@1 异常偏低，根因是
① 自包含 ViTBlock 前向的预归一化顺序写错（已修正为 <code>x = x + mlp(norm2(x))</code>），
② <code>auroc_rank_based</code> 计数方向反了（已修正为统计 neg&lt;pos）。修正后指标恢复至上述真实水平，说明评测链路可信。</div>

<p class="sub">生成时间（UTC）：{m['generated_at_utc']}</p>
</div></body></html>"""

    (REP / "index.html").write_text(html, encoding="utf-8")
    print(f"[saved] {REP / 'index.html'} ({len(html)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
