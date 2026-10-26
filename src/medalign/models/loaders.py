"""Weight loading and fingerprinting for the requested model.

Why this module is short and has **no fallbacks**: every "reasonable alternative"
(FastAI / Manual dataset / whatever the filesystem happens to contain) is exactly
how unverifiable results sneak in. Here there is one entry point per model,
and it either finds the exact expected tensor dict or raises.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..dataio import PROJECT_ROOT, human_readable_size, read_json, sha256_file, write_json
from .paths import find_one_file
from .spec import get_spec

_MASKEDPROTOTYPE_KEYS = ("partial.bin", "partial.safetensors")


@dataclass(frozen=True)
class ModelFingerprint:
    """Identity of the weights actually loaded.

    file_sha256 hashes the bytes on disk (including container metadata).
    tensor_sha256 hashes only what the tensors *are* (key, dtype, shape, bytes),
    so re-saving the same weights in another container keeps this stable.
    """

    model_id: str
    weights_path: str
    file_sha256: str
    tensor_sha256: str
    n_tensors: int
    n_parameters: int

    @property
    def model_small(self) -> str:
        return human_readable_size(self.n_parameters * 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "weights_path": self.weights_path,
            "file_sha256": self.file_sha256,
            "tensor_sha256": self.tensor_sha256,
            "n_tensors": self.n_tensors,
            "n_parameters": self.n_parameters,
            "model_small": self.model_small,
            "model_big": human_readable_size(self.n_parameters * 4),
        }


@dataclass(frozen=True)
class LoadedWeights:
    """The resolved weights together with where they came from."""

    model_id: str
    state_dict: dict[str, torch.Tensor]
    fingerprint: ModelFingerprint

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"model_id": self.model_id}
        payload.update(self.fingerprint.to_dict())
        return payload


@dataclass(frozen=True)
class LoadedModel:
    """Weights already bound to a runnable module with the right processors."""

    model_id: str
    model: Any
    state_dict: dict[str, torch.Tensor]
    fingerprint: ModelFingerprint
    processor: Any
    tokenizer: Any = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"model_id": self.model_id}
        payload.update(self.fingerprint.to_dict())
        return payload


def _reject_if_masked(weights_path: Path) -> None:
    """Reject known-bad corrupted checkpoints before anything else."""
    if weights_path.name.endswith(_MASKEDPROTOTYPE_KEYS):
        raise ValueError(
            f"{weights_path} looks like a partial download ({_MASKEDPROTOTYPE_KEYS}); "
            "re-download the full weights and try again."
        )


def _resolve_weights_path(model_id: str) -> Path:
    spec = get_spec(model_id)
    candidates: list[str] = []
    if spec.local_path:
        candidates.append(spec.local_path)
    candidates.append(str(PROJECT_ROOT / "data" / "external" / "models" / spec.family))

    found = find_one_file(candidates, f"model:{model_id}")
    root = found.root
    if root is None:
        searched = "\n  ".join(found.hits.keys())
        raise FileNotFoundError(
            f"No model directory found for '{model_id}'. "
            f"Searched:\n  {searched}\n"
            f"Run scripts/run_p0_build_specs.py with the weights present."
        )

    weights_file = spec.weights_file
    if not weights_file:
        raise ValueError(f"'{model_id}' does not declare a weights_file; cannot load weights")

    weights_path = Path(weights_file)
    if not weights_path.is_absolute():
        weights_path = root / weights_file
    _reject_if_masked(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model dir {root} exists but weights {weights_file} were not found there."
        )
    return weights_path


def _read_raw_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))

    # torch checkpoint; some failures come from an old weights_only default.
    checkpoint = torch.load(str(path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    return {k: v for k, v in state_dict.items() if hasattr(v, "detach")}


def _pre_torch_dict(
    state_dict: Mapping[str, torch.Tensor], prefix: str | None
) -> dict[str, torch.Tensor]:
    if prefix is None:
        return dict(state_dict)
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def tensor_digest(state_dict: Mapping[str, torch.Tensor]) -> tuple[str, int, int]:
    """Hash what the tensors *are* — key, dtype, shape, and raw bytes."""
    h = hashlib.sha256()
    n_params = 0
    for key in sorted(state_dict):
        arr = state_dict[key].detach().cpu().numpy()
        h.update(key.encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(",".join(map(str, arr.shape)).encode("utf-8"))
        h.update(arr.tobytes())
        n_params += int(arr.size)
    return h.hexdigest(), len(state_dict), n_params


def load_weights(
    model_id: str, prefix: str | None = None, check_missing: bool = True
) -> LoadedWeights:
    """Read the exact expected checkpoint and fingerprint it.

    prefix filters/strips when a spec stores several models in one file.
    No path guessing beyond what the model lock declares: either the file is
    there or this fails loudly.
    """
    weights_path = _resolve_weights_path(model_id)
    raw = _read_raw_checkpoint(weights_path)
    state_dict = _pre_torch_dict(raw, prefix)

    if not state_dict:
        raise ValueError(f"{weights_path} contains no tensors for prefix={prefix!r}")

    missing = _missing_keys(model_id, state_dict) if check_missing else []
    if missing:
        raise KeyError(
            f"'{model_id}' expected keys missing from {weights_path}: {missing}\n"
            "The checkpoint does not match what this model id claims to be."
        )

    file_hash = sha256_file(weights_path)
    tensor_hash, n_tensors, n_params = tensor_digest(state_dict)
    fingerprint = ModelFingerprint(
        model_id=model_id,
        weights_path=str(weights_path),
        file_sha256=file_hash,
        tensor_sha256=tensor_hash,
        n_tensors=n_tensors,
        n_parameters=n_params,
    )
    return LoadedWeights(model_id=model_id, state_dict=state_dict, fingerprint=fingerprint)


def _missing_keys(model_id: str, state_dict: Mapping[str, torch.Tensor]) -> list[str]:
    spec = get_spec(model_id)
    expected = list(spec.source_layers_manifest or [])
    if not expected:
        return []
    present = set(state_dict)
    return [k for k in expected if k not in present]


def _strict_load(model: Any, state_dict: Mapping[str, torch.Tensor], model_id: str) -> None:
    """Bind weights to a module with zero tolerance for missing/unexpected keys."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = list(missing)
    unexpected = list(unexpected)
    if missing or unexpected:
        preview_missing = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        preview_unexpected = ", ".join(unexpected[:8]) + (" ..." if len(unexpected) > 8 else "")
        raise KeyError(
            f"State dict does not exactly fit '{model_id}'.\n"
            f"  missing   ({len(missing)}): {preview_missing}\n"
            f"  unexpected({len(unexpected)}): {preview_unexpected}\n"
            "A partially loaded model silently invents values for the gap."
        )


def build_model_and_processor(model_id: str):
    """Create the empty module plus preprocessor / tokenizer for this model."""
    spec = get_spec(model_id)
    if spec.family != "open_clip" or spec.source_id != "hf-hub:Microsoft":
        raise ValueError(
            f"'{model_id}' (family={spec.family}, source={spec.source_id}) has no loader yet."
        )
    from open_clip import create_model_and_transforms, get_tokenizer

    wrapper["last_call"] = ("open_clip.create_model_and_transforms", model_id)
    model, _, processor = create_model_and_transforms(model_id, pretrained=None, device="cpu")
    tokenizer = get_tokenizer(model_id)
    return model, processor, tokenizer


wrapper: dict[str, Any] = {}


def load_model(model_id: str) -> LoadedModel:
    """Full name → real weights bound to a module. The canonical entry point."""
    spec = get_spec(model_id)
    weights = load_weights(model_id, prefix=spec.from_HF)
    model, processor, tokenizer = build_model_and_processor(model_id)
    _strict_load(model, weights.state_dict, model_id)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return LoadedModel(
        model_id=model_id,
        model=model,
        state_dict=weights.state_dict,
        fingerprint=weights.fingerprint,
        processor=processor,
        tokenizer=tokenizer,
    )


def fingerprint_to_disk(model_id: str, out_dir: str | Path | None = None) -> Path:
    """Persist the fingerprint beside cached embeddings for later cross-checks."""
    weights = load_weights(model_id)
    out_dir = Path(out_dir or (PROJECT_ROOT / "data" / "cache" / "fingerprint"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{model_id.replace('/', '__')}.json"
    write_json(out, weights.fingerprint.to_dict())
    return out
