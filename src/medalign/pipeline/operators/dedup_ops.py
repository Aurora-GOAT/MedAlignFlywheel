"""近重复识别与跨集合泄漏检查。

这一步同时服务两个目标：
1. **P1 去重**：防止候选库里的近似副本让检索指标虚高（同一张图的两个副本互为"邻居"时，
   Recall@K 会被人为抬高）。
2. **P0 泄漏检查**：检查相同字节、近似图像、高度相似图注是否跨越了官方 train/val/test 边界。

方法全部是**可复现的经典做法**（SHA256 精确匹配 / dHash·pHash 汉明距离 / token Jaccard），
不使用任何学习模型，便于第三方核对。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..operators.base import REJECT, WARN, Finding, OpResult, Operator
from ...common.perceptual import hamming
from ...common.textnorm import containment, jaccard, token_set

OP_VERSION = "dedup-leakage-v1.1"

DEFAULT_THRESHOLDS = {
    "hamming_max": 6,          # 64bit 汉明距离 ≤6 视为同一图像的不同版本
    "text_jaccard_min": 0.80,  # 词集合 Jaccard ≥0.80 视为高度相似图注
    "text_containment_min": 0.90,  # **双向**包含率都要达标，且两侧都不太短
    "text_min_tokens_for_containment": 8,
}


# ------------------------------------------------------------------
# 并查集
# ------------------------------------------------------------------
class DSU:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # 路径压缩
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # 保持字典序作为根，保证结果确定、可复现
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for node in self.parent:
            out[self.find(node)].append(node)
        return {k: sorted(v) for k, v in out.items()}


# ------------------------------------------------------------------
# 图相似度
# ------------------------------------------------------------------
def group_by_exact_hash(records: Sequence[dict], key: str = "image_sha256") -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for r in records:
        val = r.get(key)
        if val:
            out[val].append(r["sample_id"])
    return {k: sorted(v) for k, v in out.items() if len(v) > 1}


def near_duplicate_pairs_by_hamming(
    hashes: dict[str, int], hamming_max: int = 6
) -> list[tuple[str, str, int]]:
    """O(n^2) 两两比较。本项目 n 在千级及以下，足够。

    若未来规模上升，应改为按 phash 的局部敏感哈希分桶（此处留 TODO）。
    """
    keys = sorted(hashes)
    pairs: list[tuple[str, str, int]] = []
    for i in range(len(keys)):
        hi = hashes[keys[i]]
        for j in range(i + 1, len(keys)):
            d = hamming(hi, hashes[keys[j]])
            if d <= hamming_max:
                pairs.append((keys[i], keys[j], d))
    return pairs


def near_duplicate_pairs_by_text(
    tokens: dict[str, set],
    jaccard_min: float = 0.80,
    containment_min: float = 0.90,
    min_tokens_for_containment: int = 8,
) -> list[tuple[str, str, float]]:
    """先按首词倒排索引剪枝，再做两两比较，避免 O(n^2) 全量。

    关于 containment 的取值方向（这里踩过坑，值得写下来）：
    早期版本用 `max(containment(a,b), containment(b,a)) >= 0.95`，
    结果任何**短图注**只要其词全部出现在某条**长图注**里就被判为近重复，
    于是 "Chest X-ray ..." 这种模板句式把几十条完全不同的病例粘成一个 43 人的巨簇，
    进而被当成跨集合泄漏整簇剔除 —— 白白丢掉大量真实数据。

    改成 **min(containment)**（双向都要高）+ 长度下限之后，
    "一方包含另一方"不再成立，只有"两个集合几乎相同"才算近重复。
    """
    inv: dict[str, list[str]] = defaultdict(list)
    for sid, toks in tokens.items():
        for t in toks:
            inv[t].append(sid)
    candidate: set[tuple[str, str]] = set()
    for peers in inv.values():
        if len(peers) > 500:  # 极端高频词（如 "the"）跳过倒排扩散
            continue
        ps = sorted(peers)
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                candidate.add((ps[i], ps[j]))
    pairs: list[tuple[str, str, float]] = []
    for a, b in sorted(candidate):
        ja = jaccard(tokens[a], tokens[b])
        if ja >= jaccard_min:
            pairs.append((a, b, ja))
            continue
        # 双向都要几乎互相覆盖，且两边都不能太短（短图注天然容易被"包含"）
        if (
            len(tokens[a]) >= min_tokens_for_containment
            and len(tokens[b]) >= min_tokens_for_containment
            and min(containment(tokens[a], tokens[b]), containment(tokens[b], tokens[a]))
            >= containment_min
        ):
            pairs.append((a, b, ja))
    return pairs


# ------------------------------------------------------------------
# 汇总结果
# ------------------------------------------------------------------
def star_cluster(pairs: list[tuple[str, str, float]], nodes: Iterable[str]) -> dict[str, list[str]]:
    """Leader 星型聚类（canonical leader clustering）。

    为什么不用并查集做传递闭包：
    传递闭包会把"A 像 B、B 像 C、C 像 D"串成一个大团，即使 A 和 D 毫无关系。
    在医学图注上这个副作用极其严重——大量图注共享
    "MRI brain with contrast"、"Chest X-ray showed opacities in both lung fields"
    这类模板句式，传递闭包会把几十条**完全合法的病例**粘成一簇，
    进而被误判为跨集合泄漏而错误剔除，既污染实验又把本该保留的数据丢掉。

    星型聚类保证：簇内每个成员都与 leader 直接满足阈值，
    因此簇的语义是"围绕同一个代表样本的重复"，而不是"任意长度的相似链"。
    """
    # 注意：pair_set 是这里的**正确性关键**。
    # 朴素的 leader 聚类会把 "a 与 b 相似" 直接等价于 "a 加入 b 所在的簇"，
    # 于是 a 会被塞进一个它跟 leader 毫无关系的簇里（实测出现过 jaccard=0.04
    # 的 pair 被并进同一簇，进而被误判为跨集合泄漏）。
    # 这里要求：并入前必须确认该成员与 leader **本身**也满足阈值。
    pair_set = {(min(a, b), max(a, b)) for a, b, _score in pairs}
    leader_of: dict[str, str] = {}

    def linked(x: str, y: str) -> bool:
        return (min(x, y), max(x, y)) in pair_set

    for a, b, _score in sorted(pairs, key=lambda p: (p[0], p[1])):
        la, lb = leader_of.get(a), leader_of.get(b)
        if la is None and lb is None:
            leader, follower = min(a, b), max(a, b)
            leader_of[leader] = leader
            leader_of[follower] = leader
        elif la is None and linked(a, lb):
            leader_of[a] = lb
        elif lb is None and linked(b, la):
            leader_of[b] = la
        # 都不满足 → 不并入，避免形成"相似链"
    groups: dict[str, list[str]] = defaultdict(list)
    for node in sorted(nodes):
        if node in leader_of:
            groups[leader_of[node]].append(node)
    return {k: v for k, v in groups.items() if len(v) > 1}


@dataclass
class DuplicateCluster:
    cluster_id: str
    kind: str  # exact-bytes | near-image | near-caption
    members: list[str] = field(default_factory=list)
    crosses_split: bool = False
    evidence: dict = field(default_factory=dict)

    def keeper(self) -> str:
        """保留策略：字典序最小者。确定性优先于"哪张更好"，便于审计。"""
        return min(self.members)


def build_clusters(
    records: Sequence[dict], thresholds: dict | None = None
) -> list[DuplicateCluster]:
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    by_id = {r["sample_id"]: r for r in records}
    split_of = {r["sample_id"]: r.get("split", "?") for r in records}

    all_ids = sorted({r["sample_id"] for r in records})
    clusters: list[DuplicateCluster] = []

    # 1) 完全相同的字节（等价关系，用 DSU 是安全的）
    dsu_bytes = DSU()
    for sid in all_ids:
        dsu_bytes.add(sid)
    exact = group_by_exact_hash(records)
    for h, members in sorted(exact.items()):
        for m in members[1:]:
            dsu_bytes.union(members[0], m)
        clusters.append(
            DuplicateCluster(f"bytes:{h[:12]}", "exact-bytes", members,
                             len({split_of[m] for m in members}) > 1,
                             {"sha256": h})
        )
    for root, members in sorted(dsu_bytes.groups().items()):
        if len(members) > 1 and not any(
            c.kind == "exact-bytes" and set(c.members) == set(members) for c in clusters
        ):
            clusters.append(
                DuplicateCluster(f"bytes:{root}", "exact-bytes", members,
                                 len({split_of[m] for m in members}) > 1, {})
            )

    # 2) 近似图像（leader 星型聚类，不做传递闭包）
    hashes = {
        r["sample_id"]: int(r["metrics"]["phash64"])
        for r in records
        if r.get("metrics", {}).get("phash64") is not None
    }
    img_pairs = near_duplicate_pairs_by_hamming(hashes, t["hamming_max"])
    groups_img = star_cluster(img_pairs, all_ids)
    for root, members in sorted(groups_img.items()):
        evidence = {
            "pair_distances": {
                f"{a}|{b}": d for a, b, d in img_pairs if a in members and b in members
            }
        }
        clusters.append(
            DuplicateCluster(f"img:{root}", "near-image", members,
                             len({split_of[m] for m in members}) > 1, evidence)
        )

    # 3) 高度相似图注（同样用 leader 星型聚类）
    tokens = {
        r["sample_id"]: token_set(r.get("caption", ""))
        for r in records
        if r.get("caption")
    }
    txt_pairs = near_duplicate_pairs_by_text(
        tokens, t["text_jaccard_min"], t["text_containment_min"]
    )
    groups_txt = star_cluster(txt_pairs, all_ids)
    for root, members in sorted(groups_txt.items()):
        evidence = {
            "pair_jaccard": {f"{a}|{b}": round(ja, 4)
                             for a, b, ja in txt_pairs if a in members and b in members}
        }
        clusters.append(
            DuplicateCluster(f"cap:{root}", "near-caption", members,
                             len({split_of[m] for m in members}) > 1, evidence)
        )

    return clusters


class DuplicateLeakageOperator(Operator):
    """把 clusters 转成逐样本 Finding。

    两类"重复"必须区别对待，这是本算子最重要的设计决定：

    - **图像重复**（相同字节 / pHash 近似）：直接让检索候选库里出现副本，
      会把 Recall@K 人为抬高，因此必须剔除：
      * 簇内只保留字典序最小的代表，其余 reject；
      * 若簇跨 split，则整簇 reject（无法判断谁"先来"，留任何一份都会污染 test）。

    - **图注高度相似但图像不同**：这是 ROCOv2 的真实特性（大量模板化图注），
      它们是**不同的合法病例**。把它们剔除会白白丢掉 30%+ 的数据，
      而且并不能消除问题——真正会伤害评测的是"把这些 pair 当成确定负例"。
      所以这类只打 WARN 标记，样本照留，
      由 P2 的负例净化（eval/retrieval.classify_negatives）在统计负例时排除它们。
    """

    name = "dedup_leakage"
    version = OP_VERSION
    proxy_only = True

    IMAGE_KINDS = ("exact-bytes", "near-image")

    def __init__(self, clusters: Sequence[DuplicateCluster] | None = None):
        self.clusters = clusters or []

    def run(self, record: dict, ctx: dict) -> OpResult:
        sid = record["sample_id"]
        res = OpResult(sample_id=sid)
        hits = [c for c in self.clusters if sid in c.members]
        if not hits:
            res.metrics["dup_cluster_ids"] = []
            return res
        res.metrics["dup_cluster_ids"] = [c.cluster_id for c in hits]
        split_of = ctx.get("split_of", {})
        for c in hits:
            splits = sorted({split_of.get(m, "?") for m in c.members})
            if c.kind in self.IMAGE_KINDS:
                if c.crosses_split:
                    res.findings.append(
                        Finding(self.name, self.version, REJECT, "LEAKAGE_CROSS_SPLIT",
                                "图像层面的重复跨越了官方划分，整簇剔除以免检索指标虚高",
                                evidence={"cluster_id": c.cluster_id, "kind": c.kind,
                                          "members": c.members[:6], "splits": splits})
                    )
                elif sid != c.keeper():
                    res.findings.append(
                        Finding(self.name, self.version, REJECT,
                                f"DUP_{c.kind.upper().replace('-', '_')}",
                                "同 split 内的重复图像，仅保留确定性代表样本",
                                evidence={"cluster_id": c.cluster_id, "keeper": c.keeper()})
                    )
                else:
                    res.findings.append(
                        Finding(self.name, self.version, WARN,
                                f"DUP_KEEPER_{c.kind.upper().replace('-', '_')}",
                                "作为重复图像簇的代表样本保留",
                                evidence={"cluster_id": c.cluster_id, "n_members": len(c.members)})
                    )
            else:
                # near-caption：只标记，不剔除
                if c.crosses_split:
                    res.findings.append(
                        Finding(self.name, self.version, WARN, "CROSS_SPLIT_CAPTION_SIMILAR",
                                "图注与另一划分的样本高度相似，但图像不同：保留样本，"
                                "由 P2 在负例统计中把该 pair 判为 ambiguous 排除",
                                evidence={"cluster_id": c.cluster_id, "splits": splits,
                                          "n_members": len(c.members)})
                    )
                else:
                    res.findings.append(
                        Finding(self.name, self.version, WARN, "NEAR_CAPTION_SIMILAR",
                                "图注与其他样本高度相似（图像不同），保留但标记",
                                evidence={"cluster_id": c.cluster_id, "n_members": len(c.members)})
                    )
        return res
