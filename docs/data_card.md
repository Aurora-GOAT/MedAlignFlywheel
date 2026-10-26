# 数据卡 · MedAlignFlywheel

## 1. 数据集概览

| 字段 | 值 |
|---|---|
| 数据集 | ROCOv2-radiology（`eltorio/ROCOv2-radiology`） |
| DOI | 10.57967/hf/3489 |
| 官方版本 | repository Revision 5d66908（按 ROCOv2 官方 README） |
| 访问方式 | HuggingFace datasets-server `/rows` API（ROCOv2 README 推荐入口） |
| 许可 | **CC BY-NC-SA 4.0**（来源图像：PMC Open Access Subset，CC BY / CC BY-NC per upstream） |
| 计算字段来源 | caption / concept 由 ROCOv2 作者用 MedCAT v1.10.0 抽取 |
| 数据性质 | 开放获取医学论文的图注 + 图像，附 CUI 概念；**非医院真实诊断报告** |
| 再分发 | 仅本地评估，不对外重新分发整套图像 |

## 2. 摄取规模与抽样

| 项 | 值 |
|---|---|
| 上游 train 总行数 | 59,962 |
| 本项目的抽样策略 | 分块随机抽样（train 20 块×30 / val 8 块×25 / test 8 块×25） |
| 请求行数 | 1,000 |
| 实际获取唯一样本 | 1,000（源内无重复） |
| 下载图像 | 1,000（失败 0） |
| 种子 | 20260701 |

> 严格遵守"不下载完整 18GB 数据"，仅按需行级抽样并本地缓存。

## 3. 切分与清洁（P0 + P1）

| Split | 摄取 | 清洁后保留 | 拒绝 |
|---|---|---|---|
| train | 600 | 592 | 8 |
| validation | 200 | 198 | 2 |
| test | 200 | 200 | 0 |
| **合计** | **1,000** | **990** | **10** |

**拒绝原因（最终，10 条真实不合格）**：`CAPTION_TOO_SHORT` 6，`DUP_NEAR_IMAGE` 4。
**警告（不影响 keep，但进入审计）**：`CAPTION_SHORT` 44、`NEAR_CAPTION_SIMILAR` 18、`CONCEPT_INVALID_FORMAT` 14、`CAPTION_REPETITIVE` 6、`CROSS_SPLIT_CAPTION_SIMILAR` 6、`DUP_KEEPER_NEAR_IMAGE` 3。

> ⚠️ 历史更正：近重复聚类曾因传递闭包链式放大误报 **71 条跨集泄漏**，已通过"星型聚类 + 成员须与 leader 直接满足相似阈值"修复，假泄漏降为 0，最终仅剔除上述 10 条。

## 4. 跨集泄漏检查

- 近重复聚类共 **15** 个：跨切分 **3** 个（near-image 3、near-caption 0），切分内 12 个（near-caption 12）。
- 跨切分簇按"同一样本不得跨越 train/val/test"规则处理；修复后确认 **0 条假泄漏**。
- `CROSS_SPLIT_CAPTION_SIMILAR` 警告 6 条：为同一病例不同切面、图注高度相似，已标记但不强制删除（属合法近似，非泄漏）。

## 5. 模态与概念分布（清洁集 990）

**模态代理**（基于关键词，非金标）：unknown 309、CT 302、X-ray 181、MR 120、ultrasound 67、nuclear-medicine 8、photograph-clinical 5、histopathology 4、multimodal-fusion 3、graphic-chart 1。

**CUI 概念**：覆盖 **18** 个唯一 CUI，其中单例（singleton）1 个。概念稀疏，长尾效应明显——这正是 P3 长尾优先策略（D）的发力点。

## 6. 数据血缘（每条样本）

`clean_manifest.jsonl` 每条记录含：`sample_id`、`source_split_url`、`image_sha256`、`image_relpath`、`caption`、`cui`、`metrics`（phash64 / dhash64 / 长宽 / 灰度统计 / CUI 支持度）、`ops`（各算子结论）、`reject_codes` / `warn_codes`、`decision`。

## 7. 已知限制

- 图注来自论文图注，措辞模板化、同质度高，是检索 R@1 偏难（但仍达 0.8）的结构性原因。
- 模态为关键词代理，非人工金标；概念字段依赖上游 MedCAT，存在 `CONCEPT_INVALID_FORMAT` 噪声。
- 子集规模小（990），统计区间来自 Bootstrap，外推需谨慎。
- test 为冻结集，仅用于报告最终指标。
