"""文本归一化与轻量文本相似度。

设计原则：**只做可逆/可解释的字面处理**。
这里的任何规则都只是"数据质量代理指标"，不代表医学正确性。
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

_NON_PRINTABLE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_WS = re.compile(r"\s+")
# 常见的排版/占位噪声：图注里出现通常意味着抽取过程出错
# 注意：必须加词边界 \b。早期版本漏了 \b，导致 "reso**nan**ce" /
# "preg**nan**cy" / "predomi**nan**tly" 里的 nan 子串被误判为占位符，
# 一次性错杀了 47 条真实图注（见 DEVELOPMENT_PROGRESS.md 的 P1 复盘）。
_PLACEHOLDER = re.compile(
    r"\b(lorem ipsum|todo|placeholder|nan|none|null|xxx+|tbd)\b"
    r"|</?[a-z]+>|&[a-z]+;",
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-z0-9]+")

# ROCOv2 图注里常见的"面板标记"，本身不含语义信息
_PANEL_MARKERS = re.compile(
    r"\b(?:panel|figure|fig)\.?\s*[a-z0-9()\[\],\.\-–]*", re.IGNORECASE
)


def unicode_normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def normalize_caption(text: str) -> str:
    """可逆性较弱但可解释的归一化：用于去重与统计，不覆盖原始文本。"""
    t = unicode_normalize(text or "")
    t = t.replace("–", "-").replace("—", "-")
    t = _NON_PRINTABLE.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    return t


def word_tokens(text: str) -> list[str]:
    return _WORD.findall(normalize_caption(text).lower())


def token_set(text: str) -> set[str]:
    return set(word_tokens(text))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def containment(a: set[str], b: set[str]) -> float:
    """a 有多少比例被 b 覆盖（用于检测高度相似的近重复图注）。"""
    if not a:
        return 1.0
    return len(a & b) / len(a)


def char_length(text: str) -> int:
    return len(normalize_caption(text))


def nonascii_ratio(text: str) -> float:
    t = normalize_caption(text)
    if not t:
        return 0.0
    return sum(1 for ch in t if ord(ch) > 127) / len(t)


def digit_ratio(text: str) -> float:
    t = normalize_caption(text)
    if not t:
        return 0.0
    return sum(1 for ch in t if ch.isdigit()) / len(t)


def has_placeholder(text: str) -> bool:
    return bool(_PLACEHOLDER.search(text or ""))


def has_nonprintable(text: str) -> bool:
    return bool(_NON_PRINTABLE.search(text or ""))


def strip_panel_markers(text: str) -> str:
    """移除 "(a)"、"Panel B:" 之类结构标记，保留医学描述部分。"""
    t = normalize_caption(text)
    t = _PANEL_MARKERS.sub(" ", t)
    return _WS.sub(" ", t).strip()


def caption_fingerprint(text: str) -> str:
    """用于完全相同/高度相似图注的快速聚类键。"""
    toks = word_tokens(text)
    return " ".join(sorted(set(toks)))


def term_frequencies(texts: list[str], top_k: int = 50) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for t in texts:
        counter.update(set(word_tokens(t)))
    return counter.most_common(top_k)


# --------------------------------------------------------------------------
# 轻量编辑距离（不需要外部依赖），用于近重复图注的二次确认
# --------------------------------------------------------------------------
def levenshtein(a: list[str], b: list[str], max_dist: int | None = None) -> int:
    """词级 Levenshtein 距离；max_dist 用于提前剪枝，返回 max_dist 表示"更远"。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        best = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            val = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            cur.append(val)
            if val < best:
                best = val
        if max_dist is not None and best > max_dist:
            return max_dist
        prev = cur
    if max_dist is not None and prev[-1] > max_dist:
        return max_dist
    return prev[-1]
