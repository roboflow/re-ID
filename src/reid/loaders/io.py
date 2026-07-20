# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Checkpoint path resolution, state-dict loading, and load-plan resolution."""

from __future__ import annotations

import os
import pickle
import tempfile
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError
from safetensors.torch import load_file

if TYPE_CHECKING:
    from reid.preprocessing import ReIDPreprocessing

_HF_PREFIX = "hf://"
_GD_PREFIX = "gd://"

_DEFAULT_DROP_PREFIXES = ("classifier",)
_COMMON_KEY_PREFIXES = ("module.", "model.", "encoder.")


def _cache_root() -> str:
    """Return the re-ID cache root, honoring ``REID_CACHE_DIR`` when set."""
    env = os.environ.get("REID_CACHE_DIR")
    if env:
        return os.path.expanduser(env)
    return os.path.join(os.path.expanduser("~"), ".cache", "reid")


def _weights_cache_dir() -> str:
    """Return the directory used to cache downloaded weight files."""
    return os.path.join(_cache_root(), "weights")


@dataclass
class KeyReport:
    """Summary of a state-dict load (matched/missing/unexpected keys)."""

    matched: int
    total: int
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)

    @property
    def matched_fraction(self) -> float:
        return self.matched / self.total if self.total else 0.0

    def summary(self) -> str:
        return (
            f"loaded {self.matched}/{self.total} params "
            f"({self.matched_fraction:.0%}); "
            f"{len(self.missing)} missing, {len(self.unexpected)} unused"
        )


def resolve_weights(source: str) -> str:
    """Resolve a local path, ``hf://``, or ``gd://`` URL to a local weights file."""
    if source.startswith(_HF_PREFIX):
        return _resolve_hf(source)
    if source.startswith(_GD_PREFIX):
        return _resolve_gd(source)

    if not os.path.exists(source):
        raise FileNotFoundError(f"Weights file not found: {source}")
    return source


def _resolve_hf(source: str) -> str:
    rest = source[len(_HF_PREFIX) :]
    parts = rest.split("/")
    if len(parts) < 3:
        raise ValueError(f"Malformed hf:// weights URL {source!r}. Expected 'hf://<org>/<name>/<filename>'.")

    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])

    revision = None
    if "@" in repo_id:
        repo_id, revision = repo_id.split("@", 1)

    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    except (HfHubHTTPError, EntryNotFoundError, OSError) as exc:
        raise RuntimeError(
            f"Failed to download Hugging Face weights from {source!r} (repo_id={repo_id!r}, filename={filename!r})."
        ) from exc


def _validate_downloaded_file(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Downloaded weights file is missing: {path}")
    if os.path.getsize(path) < 1:
        raise OSError(f"Downloaded weights file is empty: {path}")
    return path


def _resolve_gd(source: str) -> str:
    """Download a Google Drive file once and cache under the re-ID weights cache."""
    rest = source[len(_GD_PREFIX) :]
    file_id, _, filename = rest.partition("/")
    if not file_id:
        raise ValueError(f"Malformed gd:// weights URL {source!r}. Expected 'gd://<file_id>/<filename>'.")
    if not filename:
        filename = f"{file_id}.pth"

    cache_dir = _weights_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, filename)
    if os.path.exists(path):
        return _validate_downloaded_file(path)

    try:
        import gdown
    except ImportError as exc:
        raise ImportError("Google Drive weights (gd://...) require gdown. Install with: pip install gdown") from exc

    fd, tmp_path = tempfile.mkstemp(prefix=f"{filename}.", suffix=".part", dir=cache_dir)
    os.close(fd)
    try:
        gdown.download(id=file_id, output=tmp_path, quiet=False)
        _validate_downloaded_file(tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return _validate_downloaded_file(path)


# ---------------------------------------------------------------------------
# Load-plan resolution
# ---------------------------------------------------------------------------


@dataclass
class LoadPlan:
    """Resolved recipe for building and (optionally) loading a backbone.

    Produced by :func:`resolve_load_plan` from a ``from_pretrained`` request
    (``source`` / ``architecture`` / ``preprocessing``). Callers build the
    architecture, load ``weights`` when present, and enforce
    ``required_match_fraction`` if set.
    """

    architecture: str | nn.Module
    weights: str | None
    preprocessing: ReIDPreprocessing
    domain_warning: str | None
    required_match_fraction: float | None


def resolve_load_plan(
    source: str | None,
    architecture: str | nn.Module | None,
    preprocessing: ReIDPreprocessing | None,
) -> LoadPlan:
    """Resolve a ``from_pretrained`` request into a concrete :class:`LoadPlan`.

    Handles the four load forms: default alias, curated alias / config directory
    or repo, bare weights file, and architecture-only build. When a curated card
    is resolved *and* ``architecture`` is explicitly overridden, the override
    wins (with a warning); the card's weights and preprocessing are retained.

    Raises:
        ValueError: For an architecture-only request without ``architecture``,
            or a bare weights file without ``architecture``.
    """
    from reid.catalog import (
        DEFAULT_MODEL,
        FASTREID_MOT17_SBS50,
        default_preprocessing_for_architecture,
        resolve_model_card,
    )

    # No source and no architecture → use the default curated alias.
    if source is None and architecture is None:
        source = DEFAULT_MODEL

    card = resolve_model_card(source) if source is not None else None

    arch_overridden = False
    if card is not None:
        if architecture is not None and architecture != card.architecture:
            arch_overridden = True
            warnings.warn(
                f"Overriding architecture {card.architecture!r} resolved for "
                f"{source!r} with {architecture!r}. The card's weights and "
                "preprocessing are still used — ensure they are compatible with "
                "the overridden architecture.",
                UserWarning,
                stacklevel=3,
            )
            resolved_arch: str | nn.Module = architecture
        else:
            resolved_arch = card.architecture
        resolved_weights = card.weights
        resolved_preprocessing = preprocessing if preprocessing is not None else card.preprocessing
        resolved_warning = card.domain_warning

    elif source is None:
        # Architecture-only build: random init, no network downloads.
        if architecture is None:
            raise ValueError("architecture is required when source is None")
        resolved_arch = architecture
        resolved_weights = None
        resolved_preprocessing = (
            preprocessing
            if preprocessing is not None
            else default_preprocessing_for_architecture(architecture if isinstance(architecture, str) else None)
        )
        resolved_warning = None

    else:
        # Bare weights file — architecture is required.
        if architecture is None:
            raise ValueError(
                f"Cannot load {source!r}: it appears to be a bare weights "
                "file but no `architecture` was provided. Pass e.g. "
                "architecture='osnet_x1_0' to specify the model "
                "architecture, or point to a directory produced by "
                "save_pretrained() for a self-describing checkpoint."
            )
        resolved_arch = architecture
        resolved_weights = source
        resolved_preprocessing = (
            preprocessing
            if preprocessing is not None
            else default_preprocessing_for_architecture(architecture if isinstance(architecture, str) else None)
        )
        resolved_warning = None

    # Curated aliases ship a known-good architecture+weights pairing, so require
    # a full match — unless the caller overrode the architecture, which voids
    # that guarantee and makes a partial load expected.
    required_fraction = None
    if not arch_overridden and source in (DEFAULT_MODEL, FASTREID_MOT17_SBS50):
        required_fraction = 1.0

    return LoadPlan(
        architecture=resolved_arch,
        weights=resolved_weights,
        preprocessing=resolved_preprocessing,
        domain_warning=resolved_warning,
        required_match_fraction=required_fraction,
    )


# ---------------------------------------------------------------------------
# State-dict loading
# ---------------------------------------------------------------------------


def load_state_dict_for_architecture(
    module: nn.Module,
    path: str,
    device: torch.device,
    architecture: str,
    *,
    warn_threshold: float = 0.5,
    required_match_fraction: float | None = None,
) -> KeyReport:
    """Load ``path`` into ``module``, remapping keys for ``architecture`` itself.

    The checkpoint key remap is resolved from ``architecture`` via
    :func:`reid.architectures.checkpoint_remap_for_architecture`; callers do not
    pass a remap.
    """
    from reid.architectures import checkpoint_remap_for_architecture

    remap = checkpoint_remap_for_architecture(architecture) if architecture else None
    report = load_state_dict_into(
        module,
        path,
        device,
        remap=remap,
        warn_threshold=warn_threshold,
    )
    if required_match_fraction is not None and report.matched_fraction < required_match_fraction:
        raise ValueError(
            f"Checkpoint {path!r} matched only {report.matched_fraction:.0%} of "
            f"{architecture!r} parameters (required >= {required_match_fraction:.0%}). "
            f"{report.summary()}"
        )
    return report


def _strip_common_prefixes(key: str) -> str:
    stripped = key
    changed = True
    while changed:
        changed = False
        for prefix in _COMMON_KEY_PREFIXES:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
                changed = True
    return stripped


def _read_state_dict(path: str, device: torch.device) -> dict:
    """Read a raw state dict from a ``.pth`` or ``.safetensors`` file."""
    if path.endswith(".safetensors"):
        return load_file(path, device=str(device))

    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    except (TypeError, pickle.UnpicklingError, RuntimeError) as exc:
        # weights_only=True rejects checkpoints with non-tensor globals (and old
        # torch lacks the kwarg entirely, raising TypeError). Fall back to the
        # unsafe path, but warn: it executes arbitrary pickle payloads.
        warnings.warn(
            f"Safe load (weights_only=True) failed for {path!r} "
            f"({type(exc).__name__}: {exc}); falling back to "
            "weights_only=False. Only load checkpoints from sources you trust.",
            UserWarning,
            stacklevel=2,
        )
        state_dict = torch.load(path, map_location=device, weights_only=False)

    for wrapper_key in ("state_dict", "model", "encoder"):
        if isinstance(state_dict, dict) and wrapper_key in state_dict and isinstance(state_dict[wrapper_key], dict):
            state_dict = state_dict[wrapper_key]
            break
    return state_dict


def load_state_dict_into(
    module: nn.Module,
    path: str,
    device: torch.device,
    *,
    drop_prefixes: tuple[str, ...] = _DEFAULT_DROP_PREFIXES,
    remap: Callable[[dict], dict] | None = None,
    warn_threshold: float = 0.5,
) -> KeyReport:
    """Load ``path`` into ``module`` by name and shape (classifier keys skipped)."""
    state_dict = _read_state_dict(path, device)

    cleaned: dict
    if remap is not None:
        cleaned = remap(state_dict)
    else:
        cleaned = {}
        for k, v in state_dict.items():
            key = _strip_common_prefixes(k)
            if any(key.startswith(p) for p in drop_prefixes):
                continue
            cleaned[key] = v

    target = module.state_dict()
    matched: dict = {}
    unexpected: list[str] = []
    for k, v in cleaned.items():
        if k in target and target[k].shape == v.shape:
            matched[k] = v
        else:
            unexpected.append(k)

    target.update(matched)
    module.load_state_dict(target)

    missing = [k for k in target if k not in matched]
    report = KeyReport(
        matched=len(matched),
        total=len(target),
        missing=missing,
        unexpected=unexpected,
    )

    if report.missing or report.unexpected:
        if report.matched_fraction < warn_threshold:
            warnings.warn(
                f"Only {report.summary()} from {path!r}. The weights likely do not "
                f"match this architecture — check the model/weights pairing.",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"Partial load from {path!r}: {report.summary()}. Missing "
                "parameters keep their initialised values; unused checkpoint "
                "keys were ignored.",
                UserWarning,
                stacklevel=2,
            )

    return report
