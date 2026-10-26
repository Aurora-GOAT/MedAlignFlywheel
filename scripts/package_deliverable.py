#!/usr/bin/env python
"""把 MedAlignFlywheel 的求职交付物打包成一个 zip（保留项目结构，排除大二进制）。

包含：README / docs / reports / scripts / src / tests / configs / assets /
      demo_package / data/metadata / 官方参考文件 / 官方模型配置
排除：原始图像（data/raw/**，约 81MB）、模型权重（*.bin，749MB）、
      向量缓存（data/cache/**）、__pycache__、.git、日志

用法：python scripts/package_deliverable.py [输出zip路径]
"""
from __future__ import annotations

import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]

INCLUDE_DIRS = [
    "docs", "reports", "scripts", "src", "tests", "configs", "assets", "demo_package",
    "data/metadata", "data/external/reference_official",
]
INCLUDE_FILES = [
    "README.md", ".gitignore",
    "data/external/models/biomedclip/open_clip_config.json",
]
EXCLUDE_SUFFIX = {".bin", ".npz", ".pyc", ".log"}
EXCLUDE_PARTS = {"__pycache__", ".git", "data/raw", "data/cache", "_synthetic"}

NOTE = """# MedAlignFlywheel · 求职交付包（打包说明）

生成时间：2026-10-27　|　Git 提交：fb367b4（最终验收轮）

## 最快上手

**双击 `demo_package/index.html`** —— 单文件自包含（1.3MB，24 张真实医学影像以 base64 内嵌），
离线可开、不依赖任何本机路径或网络。内含：项目定位、独立测试集结果表、模型保真度审计结论、
四类可复核真实案例、数据来源与许可。

## 先读这三份

| 文件 | 内容 |
|---|---|
| `docs/FINAL_AUDIT_REPORT.md` | **最终技术审计报告**：模型保真度 / 独立 Holdout / 飞轮价值 / 能力边界 |
| `docs/one_pager.md` | 一页中文介绍 + **三条简历描述** |
| `docs/interview_qa.md` | 面试知识库（42 题，含最终轮 Q35–Q42） |

其余：`docs/RESULTS_AND_FIXES.md`（实验表 + 缺陷修复 + 复现命令）、`docs/data_card.md`、
`docs/model_card.md`、`docs/preregistration.md`、`README.md`。

## 关键结果（独立测试集，1000 条从未参与开发的样本）

| 候选规模 | I2T R@1 | 95% CI | MRR | T2I R@1 | AUROC |
|---|---|---|---|---|---|
| 200 | 0.810 | [0.755, 0.860] | 0.883 | 0.810 | 0.9976 |
| 500 | 0.666 | [0.626, 0.708] | 0.782 | 0.656 | 0.9976 |
| 1000 | 0.587 | [0.556, 0.619] | 0.712 | 0.590 | 0.9970 |

## 打包内容

- `demo_package/` 单文件演示包
- `docs/` 全部文档；`reports/` 全部实验产物 JSON 与 `index.html`；`configs/frozen_final_eval.json` 冻结配置
- `src/`、`scripts/`、`tests/` 完整源码；`assets/architecture.svg` 架构图与缩略图
- `data/metadata/` 数据清单与血缘；`data/external/reference_official/` 官方词表与分词器文件（审计证据）

## 有意排除（体积/许可原因，不影响理解与复现）

- 原始图像 `data/raw/**`（约 81MB）—— 演示包已内嵌关键图像；复现时由脚本重新抓取
- 模型权重 `*.bin`（749MB，BiomedCLIP 官方权重）—— 由 `scripts/download_models.py` 重新下载
- 向量缓存 `data/cache/**`（7.9MB）—— 由 `scripts/run_p2_extract_embeddings.py` 重新生成

## 复现命令（medalign venv，torch 1.13.1+cu116）

```bash
python scripts/run_p0_ingest.py                 # 摄取公开数据
python scripts/run_p1_quality.py                # 六维质量算子
python scripts/run_p2_extract_embeddings.py --splits train validation test
python scripts/run_p2_retrieval_eval.py --splits validation test
python scripts/run_p0_fidelity_audit_v2.py      # 模型保真度静态审计
python scripts/run_p0_fetch_independent_testset.py 1000 2000   # 抓取独立测试集
python scripts/run_p0_independent_test_eval.py --device cuda   # 冻结配置→只评价一次
python scripts/run_p3_flywheel.py --budget 200
python scripts/run_p3_value_probe.py            # 零训练成本价值探针
python scripts/build_demo_package.py            # 生成单文件演示包
python tests/run_validation.py                  # 20 项自动化测试
```

## 图片许可

图像来自 ROCOv2（Hugging Face `eltorio/ROCOv2-radiology`，DOI 10.57967/hf/3489），
许可 **CC BY-NC-SA 4.0**，来源图像出自 PMC Open Access Subset。
本包仅用于**非商业**求职作品集展示；公开发布前请再次确认署名与许可要求。
"""


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT.parent / "MedAlignFlywheel_交付包_20261027.zip")
    files: list[tuple[Path, str]] = []
    for d in INCLUDE_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(ROOT).as_posix()
            if p.suffix in EXCLUDE_SUFFIX:
                continue
            if any(part in rel.split("/") for part in EXCLUDE_PARTS):
                continue
            files.append((p, rel))
    for f in INCLUDE_FILES:
        p = ROOT / f
        if p.exists():
            files.append((p, f))
    # 去重并排序
    seen, uniq = set(), []
    for p, rel in files:
        if rel in seen:
            continue
        seen.add(rel)
        uniq.append((p, rel))
    uniq.sort(key=lambda x: x[1])

    out.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(out, "w", ZIP_DEFLATED) as z:
        z.writestr("00_打包说明.md", NOTE)
        for p, rel in uniq:
            z.write(p, rel)
    total = sum(p.stat().st_size for p, _ in uniq)
    print(f"[packed] {len(uniq)} files, {total/1024/1024:.2f} MB (uncompressed)")
    print(f"[saved ] {out}  ({out.stat().st_size/1024/1024:.2f} MB compressed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
