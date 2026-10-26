"""影像模态推断（**代理标签**，不是官方标签）。

背景：HuggingFace 上的 ROCOv2 记录只有 image / image_id / caption / cui 四个字段，
**没有**官方 README 里提到的 modality / body_region 字段。
因此对模态的分析只能由规则从图注推断。

规则设计原则：
1. 只看字面关键词，可核查（每条判断都能指出命中的词）；
2. 多模态词同时出现时给出 `ambiguous` 而不是硬猜一个；
3. 所有输出都带 `rule_version`，便于复现；
4. 明确声明这是数据质量代理指标，**不代表医学影像分类真值**。
"""

from __future__ import annotations

import re
from typing import Iterable

RULE_VERSION = "modality-proxy-v1.0"

# 顺序敏感：先匹配更具体的模态，避免 "computed tomography" 被 CT 的通配形式吞掉
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("x-ray", re.compile(r"\b(x-?ray|radiograph|chest film|plain film|mammograph\w*|fluoroscop\w*|angiograph\w*|"
                         r"roentgenogram|venogram|barium (?:swallow|enema|meal)|pyelogram|tomosynthesis)\b", re.I)),
    ("computed-tomography", re.compile(r"\b(ct|c\.t\.|computed tomography|cat scan|ct scan|cta\b|ct angiograph\w*|"
                                       r"non-?contrast ct|ncct|hrct)\b", re.I)),
    ("magnetic-resonance", re.compile(r"\b(mri|m\.r\.i\.|mr imaging|magnetic resonance|mr image|t1-?weighted|"
                                      r"t2-?weighted|flair|dwi\b|dti\b|mr angiograph\w*|mra\b|mrv\b|fMRI)\b", re.I)),
    ("ultrasound", re.compile(r"\b(ultrasound|ultrasonograph\w*|sonograph\w*|doppler|echocardiograph\w*|"
                              r"elastograph\w*|b-?mode)\b", re.I)),
    ("nuclear-medicine", re.compile(r"\b(pet\b|pet/ct|pet-ct|positron emission|spect\b|single-?photon|"
                                    r"scintigraph\w*|bone scan|gamma camera|fdg)\b", re.I)),
    ("histopathology", re.compile(r"\b(histopatholog\w*|histolog\w*|immunohistochem\w*|h&e\b|h & e\b|h-e\b|"
                                  r"micrograph|cytolog\w*| biopsy|^\s*histology)\b", re.I)),
    ("photograph-clinical", re.compile(r"\b(clinical photograph|photograph of|intraoperative photo|"
                                       r"gross specimen|endoscop\w*|dermatoscop\w*|fundus|ophthalmoscop\w*)\b", re.I)),
    ("graphic-chart", re.compile(r"\b(graph(?:ic)? showing|bar chart|line chart|pie chart|flow[- ]?chart|"
                                 r"scatter plot|histogram|box plot|kaplan[- ]?meier|schematic|diagram|"
                                 r"prisma|consort)\b", re.I)),
]

# 只取unu在第二 priority 的信号：有些词在多种模态中都可能
_WEAK_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("computed-tomography", re.compile(r"\b(axial|coronal|sagittal) (?:section|image|view|plane)\b", re.I)),
    ("magnetic-resonance", re.compile(r"\b(gadolinium|diffusion[- ]weighted)\b", re.I)),
    ("x-ray", re.compile(r"\b(pa view|ap view|lateral view|chest film)\b", re.I)),
]

_MULTIMODAL_HINT = re.compile(r"\b(pet/?ct|pet/?mri|spect/?ct|fused|fusion image)\b", re.I)


def infer_modality(caption: str) -> dict:
    """返回 {'modality': str, 'matched': list[str], 'ambiguous': bool, 'rule_version': str}"""
    text = caption or ""
    hits: list[tuple[str, str]] = []
    for label, pat in _PATTERNS:
        m = pat.search(text)
        if m:
            hits.append((label, m.group(0)))
    labels = [h[0] for h in hits]
    uniq = sorted(set(labels))

    if not uniq:
        for label, pat in _WEAK_PATTERNS:
            m = pat.search(text)
            if m:
                return {
                    "modality": label,
                    "matched": [m.group(0)],
                    "ambiguous": False,
                    "source": "weak-rule",
                    "rule_version": RULE_VERSION,
                }
        return {
            "modality": "unknown",
            "matched": [],
            "ambiguous": False,
            "source": "no-rule-hit",
            "rule_version": RULE_VERSION,
        }

    if len(uniq) > 1:
        # 若同时出现两种模态，但我们知道它是融合影像，则给 "multimodal"
        if _MULTIMODAL_HINT.search(text):
            return {
                "modality": "multimodal-fusion",
                "matched": [h[1] for h in hits],
                "ambiguous": True,
                "source": "fusion-hint",
                "rule_version": RULE_VERSION,
            }
        return {
            "modality": uniq[0],
            "matched": [h[1] for h in hits],
            "ambiguous": True,
            "source": f"multiple-candidates:{','.join(uniq)}",
            "rule_version": RULE_VERSION,
        }

    return {
        "modality": uniq[0],
        "matched": [h[1] for h in hits],
        "ambiguous": False,
        "source": "strong-rule",
        "rule_version": RULE_VERSION,
    }


# --------------------------------------------------------------------------
# 解剖部位（同样是代理标签）
# --------------------------------------------------------------------------
REGION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("head-neck", re.compile(r"\b(brain|head|skull|cranial|intracranial|orbit|sinus|neck|thyroid|"
                             r"carotid|parotid|temporal|pituitary|cerebr\w*)\b", re.I)),
    ("chest", re.compile(r"\b(chest|lung|pulmonary|thoracic|mediastin\w*|cardiac|heart|pleura\w*|"
                         r"bronch\w*|trachea|mammar\w*|breast|aorta)\b", re.I)),
    ("abdomen", re.compile(r"\b(abdomen|abdominal|liver|hepatic|kidney|renal|spleen|splenic|pancrea\w*|"
                           r"gallbladder|stomach|gastric|intestin\w*|bowel|colon|rectal|peritone\w*)\b", re.I)),
    ("pelvis", re.compile(r"\b(pelvis|pelvic|bladder|prostate|uter\w*|ovar\w*|cervix|testis|scrot\w*)\b", re.I)),
    ("musculoskeletal", re.compile(r"\b(bone|bony|femur|humerus|radius|ulna|tibia|fibula|spine|spinal|"
                                   r"vertebra\w*|pelvis fracture|shoulder|knee|hip|ankle|wrist|joint|"
                                   r"ligament|meniscus|tendon|cervical spine|lumbar)\b", re.I)),
    ("extremity-vascular", re.compile(r"\b(arter\w*|vein|venous|vascular|angio\w*| aorta\b|iliac|"
                                      r"extremit\w*|arm|leg|hand|foot)\b", re.I)),
]


def infer_region(caption: str) -> dict:
    text = caption or ""
    hits: list[tuple[str, str]] = []
    for label, pat in REGION_PATTERNS:
        m = pat.search(text)
        if m:
            hits.append((label, m.group(0)))
    uniq = sorted(set(h[0] for h in hits))
    if not uniq:
        return {"region": "unknown", "matched": [], "ambiguous": False, "rule_version": RULE_VERSION}
    return {
        "region": uniq[0],
        "matched": [h[1] for h in hits],
        "ambiguous": len(uniq) > 1,
        "rule_version": RULE_VERSION,
    }


def caption_complexity(caption: str) -> dict:
    """图注复杂度代理：长度-分词数-是否含测量数值-是否含多条派生信息。"""
    text = caption or ""
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    has_number = bool(re.search(r"\d", text))
    has_unit = bool(re.search(r"\b(mm|cm|ml|mg|mmhg|gray|gy|hu|hz|%)\b", text, re.I))
    segments = [s for s in re.split(r"[.;:]\s+", text) if s.strip()]
    return {
        "n_chars": len(text),
        "n_tokens": len(tokens),
        "n_segments": max(1, len(segments)),
        "has_number": has_number,
        "has_unit": has_unit,
        "complexity_band": (
            "short" if len(tokens) <= 8 else "medium" if len(tokens) <= 20 else "long"
        ),
    }


def summarize_modality(modality_iter: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for m in modality_iter:
        out[m] = out.get(m, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
