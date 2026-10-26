# 模型卡 · BiomedCLIP（本项目使用的冻结编码器）

## 1. 模型身份

| 字段 | 值 |
|---|---|
| 官方 HF ID | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` |
| 视觉塔 | ViT-B/16（timm `vit_base_patch16_224`，patch 16，宽 768，12 层，预归一化） |
| 文本塔 | PubMedBERT-base（12 层，隐藏 768，上下文 256） |
| 投影 | 视觉 Linear(768→512)；文本 MLP(768→640→512) |
| 嵌入维度 | 512 |
| 参数量 | **195,902,721**（351 张量） |
| 本项目架构版本 | `medalign-biomedclip-torch-v1.0` |

## 2. 权重指纹（可复现审计）

| 指纹 | 值 |
|---|---|
| 权重文件 | `open_clip_pytorch_model.bin`（783,705,670 字节） |
| 文件 SHA256 | `52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be` |
| 张量级 SHA256 | `769bc1d3d7aa06aa3dec1d4acabdf77ee4544b72650a3bf838678568eba46af2` |
| 加载模式 | **strict · zero-missing · zero-unexpected** |
| 剔除的非参数 buffer | `text.transformer.embeddings.position_ids`（HF 常量，本项目用 `torch.arange` 等价生成，已留痕） |
| logit_scale | 4.4454（exp ≈ 85.23） |

## 3. 自包含实现说明（为何不用 open_clip）

- 本机 torch 锁在 1.13.1+cu116（GTX 1660 Ti，驱动 511.79），`import open_clip` 会 **segfault**；`transformers`/`timm` 新版要求 torch≥2。
- 因此按官方 `open_clip_config.json` 用 torch 原生模块重建网络，并设三道闸：零缺失零多余门禁、SHA256 指纹、真实数据行为校验。

## 4. 前向正确性校验（关键）

初版检索出现 **AUROC<0.5、R@1≈0.03** 的异常，定位为两处 bug 并已修正：

1. `ViTBlock` 预归一化顺序写错（MLP 分支误用了注意力前的归一化输入）→ 改为标准 `x = x + mlp(norm2(x))`。
2. `auroc_rank_based` 计数方向反（统计成 `neg>pos`）→ 改为统计 `neg<pos`。

修正后真实检索水平（冻结 test，200 候选）：

| 方向 | R@1 | R@5 | R@10 | MRR | AUROC(clean) |
|---|---|---|---|---|---|
| I2T | 0.805 | 0.965 | 0.990 | 0.873 | 0.997 |
| T2I | 0.810 | 0.975 | 0.990 | 0.879 | — |

## 5. 使用方式

- 向量提取：`python scripts/run_p2_extract_embeddings.py --splits train validation test`（GPU，约 1 分钟）。
- 评测：`python scripts/run_p2_retrieval_eval.py --splits validation test`。
- 向量缓存于 `data/cache/emb_*.npz`，P3 难例挖掘与 P4 对齐头直接复用，不重复推理。

## 6. 局限与边界

- 本模型仅作为**冻结特征提取器 / 检索评测器**使用；不微调其权重（P4 仅训外部小对齐头）。
- 检索指标反映"模型在 ROCOv2 同质图注上的判别力"，不代表零样本临床诊断能力。
- 所有相似度均来自真实前向，无任何仿真/伪造。
