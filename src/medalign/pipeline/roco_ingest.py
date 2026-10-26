"""ROCOv2 官方数据摄取。

数据源（官方 README 指定的入口）:
    https://huggingface.co/datasets/eltorio/ROCOv2-radiology
    DOI: 10.57967/hf/3489
    许可: CC BY-NC-SA 4.0（数据集）/ 原始图片来自 PMC Open Access Subset (CC BY / CC BY-NC)

**重要边界**：ROCOv2 的图注来自**已发表的开放获取医学论文图注**，
不是医院真实影像科的放射诊断报告。本项目只把它当作"医学论文图文配对"使用，
任何对外表述都不得改称为真实临床诊断数据。

抽样方式：**分块系统抽样**（block sampling）。
在官方 train/validation/test 划分内，先随机撒若干块起点，再在每个块内连续取若干行。
这样做的原因是：既保持官方划分不被破坏，又能让样本分散在整个 split 里，
同时保留"同一篇论文的多张图容易相邻出现"的自然聚集，便于后续发现来源聚集泄漏。
"""

from __future__ import annotations

import base64
import concurrent.futures as futures
import json
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..common.hashing import sha256_bytes, write_json, write_jsonl
from .http_util import fetch_bytes, fetch_json

DATASET_ID = "eltorio/ROCOv2-radiology"
DATASET_DOI = "10.57967/hf/3489"
DATASET_REVISION_CLAIMED = "dataset repository 'Revision 5d66908' (per ROCOv2 official README)"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
SIZE_ENDPOINT = "https://datasets-server.huggingface.co/size"

LICENSE_SUMMARY = {
    "dataset_license": "CC BY-NC-SA 4.0",
    "source_images": "PMC Open Access Subset (CC BY / CC BY-NC per upstream paper)",
    "computed_fields": "captions concepts extracted with MedCAT v1.10.0 (ROCOv2 authors)",
    "redistribution_note": (
        "NonCommercial + ShareAlike. 本项目仅下载子集用于本地评估，"
        "不对外重新分发整套图像；HTML 演示只引用本地路径与出处。"
    ),
}


def rows_url(split: str, offset: int, length: int) -> str:
    return (
        f"{ROWS_ENDPOINT}?dataset={DATASET_ID.replace('/', '%2F')}"
        f"&config=default&split={split}&offset={offset}&length={length}"
    )


@dataclass
class BlockPlan:
    """一个 split 的分块抽样计划（写入审计文件，保证可复现）。"""

    split: str
    num_rows_total: int
    block_starts: list[int]
    block_size: int
    take_per_block: int
    seed: int

    @property
    def planned_rows(self) -> int:
        return len(self.block_starts) * self.take_per_block


def plan_blocks(
    num_rows_total: int,
    n_blocks: int,
    take_per_block: int,
    seed: int,
    split: str,
) -> BlockPlan:
    """在 [0, num_rows_total) 里撒 n_blocks 个起点，每块取 take_per_block 行。"""
    rng = random.Random(seed)
    max_start = max(0, num_rows_total - take_per_block)
    if n_blocks <= 0 or max_start <= 0:
        return BlockPlan(split, num_rows_total, [], take_per_block, take_per_block, seed)
    if n_blocks == 1:
        starts = [rng.randrange(0, max_start + 1)]
    else:
        # 分层：把整个 split 切成 n_blocks 个等宽层，每层内随机取一个起点
        # 这样比纯均匀 random.sample 更能保证覆盖整个数据集的分布
        layer = max_start / n_blocks
        starts = []
        for i in range(n_blocks):
            lo = int(i * layer)
            hi = int((i + 1) * layer)
            if hi <= lo:
                hi = lo + 1
            starts.append(min(rng.randrange(lo, hi), max_start))
    return BlockPlan(
        split=split,
        num_rows_total=num_rows_total,
        block_starts=sorted(starts),
        block_size=take_per_block,
        take_per_block=take_per_block,
        seed=seed,
    )


def fetch_rows_for_plan(
    plan: BlockPlan,
    request_audit: list[dict],
    cache_dir: str | None = None,
    sleep_between: float = 2.0,
    on_block=None,
) -> list[dict]:
    """逐块拉取；启用本地块级缓存以支持断点续跑（对 429 限流很关键）。"""
    rows_out: list[dict] = []
    cache_path = Path(cache_dir) if cache_dir else None
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)
    for bi, start in enumerate(plan.block_starts):
        cache_file = (
            cache_path / f"{plan.split}__off{start}__len{plan.take_per_block}.json"
            if cache_path
            else None
        )
        if cache_file and cache_file.exists():
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            cached = True
        else:
            url = rows_url(plan.split, start, plan.take_per_block)
            t0 = time.time()
            payload = fetch_json(url)
            elapsed = time.time() - t0
            cached = False
            if cache_file:
                cache_file.write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
            request_audit.append(
                {
                    "split": plan.split,
                    "offset": start,
                    "length": plan.take_per_block,
                    "url": url,
                    "returned": len(payload.get("rows", [])),
                    "partial": payload.get("partial"),
                    "elapsed_sec": round(elapsed, 3),
                    "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            if bi < len(plan.block_starts) - 1:
                time.sleep(sleep_between)
        page_rows = payload.get("rows", [])
        if callable(on_block):
            on_block(plan.split, start, len(page_rows), cached)
        for item in page_rows:
            row = dict(item["row"])
            row["_row_idx"] = item.get("row_idx")
            row["_split"] = plan.split
            row["_source_url"] = f"rows_api:{plan.split}:offset={start}"
            rows_out.append(row)
    return rows_out


# --------------------------------------------------------------------------
# 图像字节处理
# --------------------------------------------------------------------------
_MAGIC_EXT = [
    (b"\xff\xd8\xff", "jpg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"GIF8", "gif"),
    (b"BM", "bmp"),
    (b"II*\x00", "tif"),
    (b"MM\x00*", "tif"),
]


def sniff_ext(data: bytes) -> str:
    for magic, ext in _MAGIC_EXT:
        if data.startswith(magic):
            return ext
    return "bin"


def _download_one(args: tuple[str, str, str, float]) -> dict:
    image_id, src, dest_dir, timeout = args
    url = src
    if src.startswith("data:"):
        # datasets-server 偶尔内联 base64
        header, b64 = src.split(",", 1)
        data = base64.b64decode(b64)
    else:
        data = fetch_bytes(url, timeout=timeout, retries=3)
    ext = sniff_ext(data)
    dest = Path(dest_dir) / f"{image_id}.{ext}"
    dest.write_bytes(data)
    return {
        "image_id": image_id,
        "relative_path": dest.name,
        "bytes": len(data),
        "bytes_sha256": sha256_bytes(data),
        "ext": ext,
        "image_url": src,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def download_images(
    records: list[dict], dest_dir: str, workers: int = 8, timeout: float = 60.0
) -> tuple[list[dict], list[dict]]:
    """并发下载并写盘；返回 (成功列表, 失败列表)。失败原因也要落盘审计。

    已存在于 dest_dir 的 sample_id 直接复用（断点续跑）。
    """
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    existing: dict[str, Path] = {}
    for p in Path(dest_dir).glob("*"):
        if p.is_file():
            existing.setdefault(p.stem, p)
    tasks = []
    reused: list[dict] = []
    for rec in records:
        src = rec.get("image", {}).get("src") if isinstance(rec.get("image"), dict) else rec.get("image")
        if not (isinstance(src, str) and src):
            continue
        iid = rec["image_id"]
        prior = existing.get(iid)
        if prior is not None and prior.stat().st_size > 0:
            data = prior.read_bytes()
            reused.append(
                {
                    "image_id": iid,
                    "relative_path": prior.name,
                    "bytes": len(data),
                    "bytes_sha256": sha256_bytes(data),
                    "ext": prior.suffix.lstrip("."),
                    "image_url": src,
                    "downloaded_at_utc": "reused-from-local-cache",
                }
            )
            continue
        tasks.append((iid, src, dest_dir, timeout))
    ok: list[dict] = list(reused)
    failed: list[dict] = []
    with futures.ThreadPoolExecutor(max_workers=workers) as ex:
        fut_map = {ex.submit(_download_one, t): t[0] for t in tasks}
        for fut in futures.as_completed(fut_map):
            image_id = fut_map[fut]
            try:
                ok.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                failed.append({"image_id": image_id, "error": f"{type(exc).__name__}: {exc}"})
    return ok, failed


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
@dataclass
class IngestConfig:
    out_root: str
    seed: int = 20260701
    n_blocks_train: int = 20
    take_train: int = 30
    n_blocks_val: int = 8
    take_val: int = 25
    n_blocks_test: int = 8
    take_test: int = 25
    workers: int = 8


@dataclass
class IngestResult:
    raw_manifest: str = ""
    provenance: str = ""
    audit: str = ""
    n_rows_requested: int = 0
    n_rows_fetched: int = 0
    n_images_ok: int = 0
    n_images_failed: int = 0


def _plan_for(cfg: IngestConfig, split: str, n_blocks: int, take: int) -> BlockPlan:
    size_payload = fetch_json(f"{SIZE_ENDPOINT}?dataset={DATASET_ID.replace('/', '%2F')}")
    n_total = 0
    for s in size_payload["size"]["splits"]:
        if s["split"] == split:
            n_total = s["num_rows"]
    if not n_total:
        raise RuntimeError(f"无法获取 {split} 的行数: {size_payload}")
    return plan_blocks(n_total, n_blocks, take, cfg.seed, split)


def run_ingest(cfg: IngestConfig) -> IngestResult:
    out_root = Path(cfg.out_root)
    meta_dir = out_root / "data" / "metadata"
    img_dir = out_root / "data" / "raw" / "images"
    meta_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    request_audit: list[dict] = []
    all_rows: list[dict] = []
    plans = {}
    cache_dir = meta_dir / "_rows_cache"
    for split, nb, take in (
        ("train", cfg.n_blocks_train, cfg.take_train),
        ("validation", cfg.n_blocks_val, cfg.take_val),
        ("test", cfg.n_blocks_test, cfg.take_test),
    ):
        plan = _plan_for(cfg, split, nb, take)
        plans[split] = plan

        def _progress(sp=split, s=0, n=0, ch=False):
            print(f"[rows] {sp} offset={s} rows={n} cached={ch}", flush=True)

        rows = fetch_rows_for_plan(
            plan, request_audit, cache_dir=str(cache_dir),
            sleep_between=3.0, on_block=_progress,
        )
        all_rows.extend(rows)

    # image_id 去重（官方保证唯一，但仍显式校验）
    seen: dict[str, dict] = {}
    duplicates_in_source: list[dict] = []
    for r in all_rows:
        iid = r.get("image_id")
        if not iid:
            duplicates_in_source.append({"reason": "missing_image_id", "row_idx": r.get("_row_idx")})
            continue
        if iid in seen:
            duplicates_in_source.append(
                {"reason": "duplicate_image_id", "image_id": iid, "row_idx": r.get("_row_idx")}
            )
            continue
        seen[iid] = r

    records = list(seen.values())
    ok_images, failed_images = download_images(records, str(img_dir), workers=cfg.workers)
    ok_by_id = {d["image_id"]: d for d in ok_images}
    failed_by_id = {d["image_id"]: d for d in failed_images}

    manifest_rows = []
    for r in records:
        iid = r["image_id"]
        img = ok_by_id.get(iid)
        if img is None:
            manifest_rows.append(
                {
                    "sample_id": iid,
                    "split": r["_split"],
                    "row_idx": r.get("_row_idx"),
                    "caption": r.get("caption", ""),
                    "cui": list(r.get("cui") or []),
                    "image_available": False,
                    "download_error": failed_by_id.get(iid, {}).get("error", "unknown"),
                    "source_split_url": r.get("_source_url"),
                }
            )
            continue
        manifest_rows.append(
            {
                "sample_id": iid,
                "split": r["_split"],
                "row_idx": r.get("_row_idx"),
                "caption": r.get("caption", ""),
                "cui": list(r.get("cui") or []),
                "image_available": True,
                "image_relpath": f"images/{img['relative_path']}",
                "image_bytes": img["bytes"],
                "image_sha256": img["bytes_sha256"],
                "image_ext": img["ext"],
                "image_upstream_url": img["image_url"],
                "source_split_url": r.get("_source_url"),
            }
        )

    manifest_path = meta_dir / "raw_manifest.jsonl"
    write_jsonl(manifest_path, manifest_rows)

    provenance = {
        "dataset_id": DATASET_ID,
        "doi": DATASET_DOI,
        "revision_claimed_in_official_readme": DATASET_REVISION_CLAIMED,
        "access_route": "HuggingFace datasets-server /rows API (official entry recommended by ROCOv2 README)",
        "license": LICENSE_SUMMARY,
        "data_nature_warning": (
            "图注来自开放获取医学论文的图注，不是医院真实影像诊断报告；不得改称。"
        ),
        "ingest_config": asdict(cfg),
        "sampling_plans": {k: asdict(v) for k, v in plans.items()},
        "n_rows_requested": sum(p.planned_rows for p in plans.values()),
        "n_rows_fetched": len(all_rows),
        "n_unique_samples": len(records),
        "duplicates_in_source": duplicates_in_source,
        "n_images_downloaded": len(ok_images),
        "n_images_failed": len(failed_images),
        "failed_images": failed_images,
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        "columns_available_in_source": ["image", "image_id", "caption", "cui"],
        "columns_absent_note": (
            "HF 版本的记录中不含独立的 modality / body_region 字段，"
            "也不含论文 PMID/PMC 链接。本项目的模态标签只能由规则从图注推断，"
            "并明确标注为数据质量代理指标（见 docs/DATA_CARD.md）。"
        ),
    }
    prov_path = meta_dir / "p0_provenance.json"
    write_json(prov_path, provenance)

    audit_path = meta_dir / "p0_requests_audit.jsonl"
    write_jsonl(audit_path, request_audit)

    return IngestResult(
        raw_manifest=str(manifest_path),
        provenance=str(prov_path),
        audit=str(audit_path),
        n_rows_requested=sum(p.planned_rows for p in plans.values()),
        n_rows_fetched=len(all_rows),
        n_images_ok=len(ok_images),
        n_images_failed=len(failed_images),
    )
