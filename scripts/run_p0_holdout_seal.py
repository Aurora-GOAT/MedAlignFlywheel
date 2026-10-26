#!/usr/bin/env python
"""P0-5：预注册一套"未被任何调参/选择/训练使用过"的未见数据，并封存其 ID 与 SHA256。

为什么需要
----------
test 切分已被反复检查并用于最终报告；为遵守"后续参数调整只能用 train/val、
最终验证用未见过的数据"的纪律，这里从 train 中确定性封存一个 holdout 子集：
- 该子集**不参与**任何超参搜索、数据选择、对齐头训练；
- 仅在"最终验证"阶段整体打开一次（且打开动作本身会被记录）；
- 封存 SHA256（ID 列表 + holdout 向量拼接），防止事后被悄悄替换。

封存规则（确定性、可复现、与任何随机种子无关）
------------------------------------------------
取 train 全量 sample_id 排序后，按 id 的 SHA1 末位 % 10 == 0 入选（约 10%）。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402


def main() -> int:
    cache = Path("data/cache")
    train_ids = json.loads((cache / "emb_train.ids.json").read_text(encoding="utf-8"))
    d = np.load(cache / "emb_train.npz", allow_pickle=True)
    img = d["image_emb"].astype(np.float32)
    txt = d["text_emb"].astype(np.float32)

    # 确定性封存：SHA1(id) 末位 % 10 == 0
    holdout_ids = sorted([i for i in train_ids if int(hashlib.sha1(i.encode()).hexdigest(), 16) % 10 == 0])
    idx_map = {i: k for k, i in enumerate(train_ids)}
    hold_idx = [idx_map[i] for i in holdout_ids]

    # 封存 SHA：ID 列表 + holdout 向量拼接
    id_blob = "\n".join(holdout_ids).encode("utf-8")
    id_sha = hashlib.sha256(id_blob).hexdigest()
    vec_blob = np.concatenate([img[hold_idx], txt[hold_idx]]).tobytes()
    vec_sha = hashlib.sha256(vec_blob).hexdigest()

    out_dir = Path("data/metadata")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "holdout_sealed.jsonl", "w", encoding="utf-8") as f:
        for i in holdout_ids:
            f.write(json.dumps({"sample_id": i, "source_split": "train", "role": "preregistered_holdout"}, ensure_ascii=False) + "\n")

    seal = {
        "sealed_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "rule": "SHA1(sample_id) mod 10 == 0 over sorted train ids (~10%)",
        "n_holdout": len(holdout_ids),
        "n_train_total": len(train_ids),
        "excluded_from": ["hyperparameter_search", "data_selection", "adapter_training", "any_tuning"],
        "open_policy": "only opened once for final verification; opening action is logged separately",
        "id_list_sha256": id_sha,
        "holdout_embeddings_sha256": vec_sha,
        "holdout_ids_sample_first20": holdout_ids[:20],
    }
    (out_dir / "holdout_seal.json").write_text(json.dumps(seal, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[sealed] n_holdout={len(holdout_ids)}/{len(train_ids)}  id_sha={id_sha[:16]} vec_sha={vec_sha[:16]}")
    print("[saved] data/metadata/holdout_sealed.jsonl  data/metadata/holdout_seal.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
