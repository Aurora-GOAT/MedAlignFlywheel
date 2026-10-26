# MedAlignFlywheel · 最新真实实验结果、缺陷修复与复现说明

> 面向蚂蚁健康「全模态大模型数据」岗位技术面试的深挖材料。
> 所有数字均来自本地真实运行（无伪造）；未实际运行的环节明确标注 `NOT_TESTED` 或 `BLOCKED`。

---

## 1 · 最新真实实验结果表

### 1.1 P2 跨模态检索（真实 BiomedCLIP 权重，冻结 test 集，200 候选）

| 指标 | test | validation（开发集） |
|---|---|---|
| I2T R@1 | **0.805** | 0.727 |
| I2T R@5 | 0.965 | — |
| I2T R@10 | 0.990 | — |
| I2T MRR | 0.873 | — |
| T2I R@1 | 0.810 | — |
| T2I MRR | 0.879 | — |
| AUROC（净化后） | **0.9975** | — |

- 正例平均相似度 0.141，净化后负例平均 0.107（二者可分）。
- 排除 **146** 个模糊负例（图注高度重叠或图像近重复），负例不被系统性高估。
- val（0.727）/ test（0.805）量级一致，**无跨集泄漏迹象**。

### 1.2 候选规模对照（验证 80.5% 是否靠"候选少"虚高）

| 候选规模 | I2T R@1 | T2I R@1 | AUROC(全部非对角) |
|---|---|---|---|
| 200 | 0.805 | 0.810 | 0.9947 |
| 500 | 0.665 | 0.680 | 0.9964 |
| 1000 | 0.590 | 0.600 | 0.9969 |

> 结论：放大候选池 R@1 自然下降，但 **AUROC 始终 ≈0.997**（模型区分正负的能力不随候选数衰减），说明 80.5% 是 200 候选口径的真实值，而非"缩小候选"的假象。

### 1.3 同源/重复/相似样本是否让任务过易（P0-4 泄漏分析）

在 200 test 池内：
- 近重复图像对：**2** 对；近相同图注对：**27** 对；共享概念混淆查询：**36** 个。
- 剔除 36 个混淆查询后 R@1 = **0.811**（不降反略升）。
- 结论：**同源论文/重复图像/相似图注并未人为变易检索任务**；即使剔除最易混淆样本，指标不降。

### 1.4 P3 数据飞轮（train，预算=200）

- 难例挖掘：**229** 条（risk_false_positive 214 / hard_positive 15），全部标记 `needs_human_review`，且全部为共享概念（图文配对歧义普遍存在）。
- 难负例构造：**4,736** 候选，其中**确定负例 1,218**（三重校验：来源非近重复 / 概念不相交 / 语义不重叠），其余标记 `needs_review`。
- 四类数据选择策略（A 随机 / B 质量 / C 质量+困难 / D 质量+长尾）的数据侧差异：C 把困难样本占比从 ~27% 提升到 83%，D 长尾优先导致模态覆盖回落。

### 1.5 P4 对齐头对照（冻结编码器 + 小对齐头，同预算 200 样本）

| 设置 | test I2T R@1 | 说明 |
|---|---|---|
| 冻结基线（不训练） | **0.805** | step-0 对齐头 = 恒等映射，等于此基线 |
| A 随机 | 0.578 ± 0.016 | 过拟合拖累 |
| B 质量 | 0.578 ± 0.016 | 与 A 一致（清洁子集质量过滤近乎空操作） |
| C 质量+困难 | 0.507 ± 0.018 | 比随机更差 |
| D 质量+长尾 | 0.527 ± 0.017 | 比随机更差 |

- 多随机种子（seed 0/1/2）下四策略均稳定低于基线，差异在噪声内。
- **诚实结论**：在已接近天花板的冻结编码器上，仅用 200 样本训练小对齐头会**过拟合并拖累**检索（0.81→0.57）；困难/长尾优先的 C/D 反而更差。数据选择增益在此规模 + 冻结设置下**不可见**，须全参微调或更大数据才显形。

### 1.6 P0-1 模型可信度审计

- 实时官方 `open_clip` 前向对比：**BLOCKED**（本机两种方案均失败：medalign/torch1.13 导入即段错误；refclip/torch2.x CPU 的 `c10.dll` 初始化失败）。
- 替代强证据链（均真实）：
  - 架构规格逐项比对（`open_clip_config.json`）：**全部匹配**（embed_dim=512、vit_base_patch16_224、vision 投影 768→512 线性、text mlp 768→640→512、cls 池化、context_length=256、预处理均值/标准差一致）。
  - 严格加载（strict zero-missing/zero-unexpected）+ 权重指纹 SHA256。
  - 前向**确定性**：同输入两次编码最大差 = 0.0。
  - 词表保真：checkpoint 文本 embedding 为 30522 行，本项目 `vocab.txt` 28895 行，但全部图注 token 的 UNK 率仅 **0.035%**（88/253,440）。
  - 检索 AUROC ≈ 0.997 佐证跨模态对齐真实存在。

### 1.7 P0-5 预注册未见数据

- 从 train（剔除已用于调参/选择的子集）中确定性封存 **55/592** 条为 holdout，记录 ID 与向量 SHA256 防替换。
- 协议：后续调参只能用 train/validation；该 holdout 仅用于**最终验证**，不参与任何训练/选择/指标选择。

---

## 2 · 实验缺陷与修复记录（公开、可复现）

| # | 现象 | 根因 | 修复 | 影响 |
|---|---|---|---|---|
| 1 | 初版检索 AUROC<0.5、R@1≈0.03 | 自包含 `ViTBlock` 前向预归一化顺序写错（注意力残差更新前误用归一化输入） | 改为标准 `x = x + mlp(norm2(x))` | 修复后 R@1→0.805、AUROC→0.997 |
| 2 | `auroc_rank_based` 返回反值（pos<neg 时得 1.0） | 计数方向写反（统计 `neg>pos` 而非 `neg<pos`） | 改为统计 `neg<pos` | AUROC 语义正确 |
| 3 | P4 初版 R@1=0.005（≈随机之下） | `AlignmentHead` "近恒等初始化"错写成输出层零初始化（前向恒为 0，破坏对齐） | 改为残差结构 `out = x + mlp(x)`，零初始化即精确恒等 | 修复后 R@1 回到 0.57（过拟合拖累的真实值，非 bug） |
| 4 | P3 跑通前脚本为坏桩 | 函数签名/字段名对不上（`selected_ids` 应为 `sample_ids`、`run_all_strategies` 缺位置参数） | 重写接线 | 飞轮真实产出 |
| 5 | 初版 `recall_at_k` 数组版被浮点版覆盖 | 同名函数重复定义 | 统一为浮点比率版并修正 `.mean()` 调用 | 分层指标正确 |
| 6 | 近重复过度合并 + 跨集泄漏误判（71 条假泄漏） | DSU 链式传递闭包放大 | 改为 star_cluster 必须校验成员与 leader 直接相似 | 假泄漏 71→0 |

> 上述 1/2 是评测链路 bug，曾被"看起来能跑"掩盖；修复后指标恢复至真实水平，证明评测链路经过真实数据校验，而非"调参调出来"。

---

## 3 · 数据来源及授权说明

- 数据集：**ROCOv2（radiology）**，Hugging Face `eltorio/ROCOv2-radiology`（DOI 10.57967/hf/3489）。
- 许可：**CC BY-NC-SA 4.0**。本项目仅用于**非商业**求职作品集展示；缩略图保留原始出处与许可信息（见 `reports/index.html` 第 5 节与 `assets/thumbnails/`）。
- 规模：原始按需抽样 **1000** 组合法图文配对 → 质量门后保留 **990**（仅 10 条被拒：6 过短图注 + 4 重复图像）。
- 切分：train 592 / validation 198 / test 200；test 冻结，不参与训练/选择/调参。
- 预训练权重：**BiomedCLIP**（`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`），195,902,721 参数，严格门禁加载。
- 红线：未下载 18GB 完整数据；未上传任何私有数据；未公开推送仓库；未调用付费 API / 租用付费 GPU。

---

## 4 · 边界严格区分（面试必答）

| 维度 | 状态 | 说明 |
|---|---|---|
| 已实现的数据质量检测 | ✅ 真实运行 | 六大质量算子 + 合成缺陷测试（5/5 自动识别、0 假阳性） |
| 已真实运行的图文检索 | ✅ 真实运行 | 真实 BiomedCLIP 前向 + 200/500/1000 候选 + 分层 + AUROC |
| 尚未经人工确认的数据错配 | ⚠️ 未确认 | `risk_false_positive` 仅表示"模型把高相似负例排在正例前"，不等同"已确诊医学错配"；每条预留 `review` 字段待人工填写 |
| 已验证 / 未验证的数据选择收益 | ⚠️ 边界结论 | P4 真实实验表明：本规模 + 冻结编码器下，A/B/C/D 选择**增益不可见**（均低于基线）；"选择有效"未被证实，亦未被证伪（须全参微调/更大数据） |
| 预训练模型能力 vs 本项目工程贡献 | 明确区分 | 0.81 的跨模态对齐来自 BiomedCLIP 预训练；本项目的**新增工程贡献**是：合规数据摄取与跨集泄漏检查、可溯源的质量算子、难例/难负例挖掘与三重校验、四种数据选择策略的可核验对比、以及"冻结编码器下 200 样本对齐头无法增益"的诚实边界结论 |

---

## 5 · 可复现命令与源码路径

```bash
# 0) 环境（medalign venv，torch 1.13.1+cu116，已装 open_clip/timm/transformers 但本机导入段错误故自实现前向）
python scripts/run_p0_ingest.py            # P0 摄取 + 跨集泄漏检查
python scripts/run_p1_quality.py           # P1 六大质量算子
python scripts/run_p2_extract_embeddings.py --splits train validation test   # GPU 提取向量
python scripts/run_p2_retrieval_eval.py --splits validation test             # P2 检索评测
python scripts/run_p2_scale_eval.py        # P0-3/0-4 候选规模 200/500/1000 + 泄漏分析
python scripts/run_p0_fidelity_audit.py    # P0-1 静态保真度审计
python scripts/run_p0_holdout_seal.py      # P0-5 预注册未见数据
python scripts/run_p3_flywheel.py --budget 200    # P3 飞轮
python scripts/run_p2_synthetic_quality.py # P2 合成缺陷测试 + A/B 诊断 + 长尾覆盖
python scripts/run_p2_hard_taxonomy.py     # P2 难例分类
python scripts/run_p3_case_gallery.py      # P3 缩略图 + 四类案例
python scripts/run_p4_diagnostic.py --device cpu   # P1/P4 诊断（step-0 恒等/梯度/曲线/多种子）
python scripts/render_html.py              # 生成 reports/index.html
python tests/run_validation.py             # 自动化验证（15 项全过）→ reports/validation_test_log.txt
```

核心源码：`src/medalign/{models, eval, flywheel, pipeline}/`；脚本：`scripts/`；产物：`reports/`；演示页：`reports/index.html`；架构图：`assets/architecture.svg`。

---

## 6 · 自动化测试日志

- 测试套件：`tests/run_validation.py`（无外部依赖，纯标准库 + 项目源码）。
- 覆盖：AUROC 方向修复、recall@k 语义、bootstrap CI、P1 对齐头 step-0 精确恒等（maxdiff=1.5e-8）、梯度数值（有限且非零）、P1 质量算子（缺失/占位/清洁无假阳性）、BiomedCLIP 严格加载（195,902,721 参数）与前向确定性（maxdiff=0.0）。
- 结果：**15/15 通过**，日志：`reports/validation_test_log.txt`。
