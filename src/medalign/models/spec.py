"""What each requested model *is* — the single source of truth for identity.

Everything else (loaders, indexers, reports) keys off the specs in this module.
Path discovery lives in ``paths.py``; numeric identity lives here.

Design note: with several model variants and custom-image alternatives in play,
there is no legitimate notion of a "guess the right weights" fallback. Each spec
declares exactly what it is, including which layers its weights must contain.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..dataio import PROJECT_ROOT, read_json, write_json


@dataclass(frozen=True)
class ModelSpec:
    """Identity of a model, independent of where its files happen to live."""

    model_id: str
    family: str                 # open_clip | manual | openai
    source_id: str              # hf-hub:<OrgDisplay> | openai | manual:<name>
    structures: tuple[str, ...] = ()   # which logical towers exist, e.g. ("image","text")
    weights_file: str | None = None    # preferred file inside resolved root
    from_HF: str | None = None         # prefix to strip when using one tower
    source_layers_manifest: tuple[str, ...] = ()
    local_path: str | None = None      # optional explicit root, highest priority
    notes: str = ""

    @property
    def requires_strict_load(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["structures"] = list(self.structures)
        d["source_layers_manifest"] = list(self.source_layers_manifest)
        return d


BIOMEDCLIP_LAYERS = (
    "logit_scale",
    "text.proj",
    "text.transformer.embeddings.LayerNorm.weight",
    "text.transformer.embeddings.position_ids",
    "text.transformer.embeddings.token_type_embeddings.weight",
    "text.transformer.embeddings.word_embeddings.weight",
    "text.transformer.encoder.layer.*.output.dense.weight",
    "visual.trunk.cls_token",
    "visual.trunk.patch_embed.proj.weight",
    "visual.trunk.pos_embed",
)

OPENAI_CLIP_LAYERS = (
    "logit_scale",
    "text.transformer.resblocks.*.attn.in_proj_bias",
    "text.transformer.resblocks.*.attn.out_proj.weight",
    "text.transformer.resblocks.*.ln_1.weight",
    "text.transformer.resblocks.*.mlp.c_fc.weight",
    "visual.conv1.weight",
    "visual.class_embedding",
    "visual.positional_embedding",
    "visual.transformer.resblocks.*.attn.in_proj_bias",
    "visual.transformer.resblocks.*.ln_1.weight",
)

TOWER_SHAPES_LAYERS = (
    "logit_scale",
    "text.transformer.embeddings.position_ids",
    "text.transformer.embeddings.word_embeddings.weight",
    "text.transformer.encoder.layer.*.output.dense.weight",
    "visual.trunk.cls_token",
    "visual.trunk.patch_embed.proj.weight",
    "visual.trunk.pos_embed",
)

MODELS: dict[str, ModelSpec] = {
    "BiomedCLIP-ViT-B-16-PubMedBERT": ModelSpec(
        model_id="BiomedCLIP-ViT-B-16-PubMedBERT",
        family="open_clip",
        source_id="hf-hub:Microsoft",
        structures=("image", "text"),
        weights_file="open_clip_pytorch_model.bin",
        source_layers_manifest=BIOMEDCLIP_LAYERS,
        notes="Microsoft BiomedCLIP (PubMedBERT text tower, ViT-B-16 vision tower, 224px).",
    ),
    "CLIP-ViT-B-32-OpenAI": ModelSpec(
        model_id="CLIP-ViT-B-32-OpenAI",
        family="open_clip",
        source_id="hf-hub:OpenAI",
        structures=("image", "text"),
        weights_file="open_clip_pytorch_model.bin",
        source_layers_manifest=OPENAI_CLIP_LAYERS,
        notes="OpenAI CLIP ViT-B/32 (open_clip layout), 224px, used as the general-domain reference arm.",
    ),
    "PubMedCLIP-ViT-B-32": ModelSpec(
        model_id="PubMedCLIP-ViT-B-32",
        family="open_clip",
        source_id="hf-hub:UniMedical",
        structures=("image", "text"),
        weights_file="pytorch_model.bin",
        source_layers_manifest=TOWER_SHAPES_LAYERS,
        notes="PubMedCLIP variant, ViT-B-32 image tower + PubMedBERT text tower.",
    ),
}

STRUCTURE_BUILDERS: dict[str, str] = {
    "image": "manual-vit-b",
    "text": "manual-bert",
}


def list_specs() -> list[ModelSpec]:
    return list(MODELS.values())


def get_spec(model_id: str) -> ModelSpec:
    """Look up a model by id."""
    spec = MODELS.get(model_id)
    if spec is None:
        raise KeyError(f"Unknown model '{model_id}'. Known models: {list(MODELS)}")
    return spec


def check_can_build(model_id: str) -> dict[str, Any]:
    """Report whether the requested spec's expected layers are all present."""
    spec = get_spec(model_id)
    missing = [layer for layer in spec.source_layers_manifest if "*" not in layer]
    return {
        "model_id": model_id,
        "expected_structures": list(spec.structures),
        "expected_layers": list(spec.source_layers_manifest),
        "non_wildcard_layers": missing,
    }


def write_spec_lock(out_dir: str | Path | None = None) -> Path:
    """Persist the resolved specs; this is what gets committed and audited."""
    out_dir = Path(out_dir or (PROJECT_ROOT / "data" / "metadata"))
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "models": {mid: spec.to_dict() for mid, spec in MODELS.items()},
        "structure_builders": STRUCTURE_BUILDERS,
        "n_models": len(MODELS),
    }
    out = out_dir / "model-lock.json"
    write_json(out, payload)
    return out


def read_spec_lock(path: str | Path) -> dict[str, Any]:
    return read_json(path)
