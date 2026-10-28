# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Checkpoint path resolution, state-dict loading, and load-plan resolution."""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import tempfile
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple
from urllib.parse import urlparse, urlunparse

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError
from safetensors.torch import load_file

if TYPE_CHECKING:
    from reid.preprocessing import ReIDPreprocessing

logger = logging.getLogger(__name__)

_HF_PREFIX = "hf://"
_GD_PREFIX = "gd://"
_URL_PREFIXES = ("http://", "https://")

_DEFAULT_DROP_PREFIXES = ("classifier",)
_COMMON_KEY_PREFIXES = ("module.", "model.", "encoder.")
_MISMATCH_FRACTION = 0.5


def _cache_root() -> str:
    """Return the re-ID cache root, honoring ``REID_CACHE_DIR`` when set."""
    env = os.environ.get("REID_CACHE_DIR")
    if env:
        return os.path.expanduser(env)
    return os.path.join(os.path.expanduser("~"), ".cache", "reid")


def _weights_cache_dir() -> str:
    """Return the directory used to cache downloaded weight files."""
    return os.path.join(_cache_root(), "weights")


def resolve_weights(source: str) -> str:
    """Resolve a local path, ``hf://``, ``gd://``, or ``http(s)://`` URL to a local weights file."""
    if source.startswith(_HF_PREFIX):
        return _resolve_hf(source)
    if source.startswith(_GD_PREFIX):
        return _resolve_gd(source)
    if source.startswith(_URL_PREFIXES):
        return _resolve_url(source)

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
            os.remove(tmp_path)
        raise

    return _validate_downloaded_file(path)


def _resolve_url(source: str) -> str:
    """Download an ``http(s)://`` file once and cache it under its host, path and query in the weights cache."""
    parsed = urlparse(source)
    # Keyed by host and path, not file name: every saved model folder holds a weights.safetensors.
    parts = parsed.path.strip("/").split("/")
    if parsed.query:
        # ?version=1 and ?version=2 are different files; the file name keeps its suffix.
        parts.insert(-1, "query-" + hashlib.sha256(parsed.query.encode()).hexdigest()[:16])
    path = os.path.join(_weights_cache_dir(), parsed.netloc, *parts)
    if os.path.exists(path):
        return _validate_downloaded_file(path)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = urlunparse(parsed._replace(fragment=""))
    try:
        torch.hub.download_url_to_file(url, path)
    except OSError as exc:
        raise RuntimeError(f"Failed to download weights from {source!r}.") from exc
    if os.path.getsize(path) == 0:
        os.remove(path)
        raise RuntimeError(f"The weights downloaded from {source!r} are empty.")
    return path


# ---------------------------------------------------------------------------
# Load-plan resolution
# ---------------------------------------------------------------------------


class LoadPlan(NamedTuple):
    """Resolved ``from_pretrained`` recipe."""

    architecture: str | nn.Module
    weights: str | None
    preprocessing: ReIDPreprocessing
    domain_warning: str | None
    required_match_fraction: float | None


def _resolve_preprocessing(
    explicit: ReIDPreprocessing | None,
    config: ReIDPreprocessing | None,
    architecture_for_default: str | nn.Module | None,
) -> ReIDPreprocessing:
    """Pick preprocessing: explicit override, then config, then architecture default."""
    from reid.architectures import default_preprocessing_for_architecture

    if explicit is not None:
        return explicit
    if config is not None:
        return config
    return default_preprocessing_for_architecture(
        architecture_for_default if isinstance(architecture_for_default, str) else None
    )


def resolve_load_plan(
    source: str | None,
    architecture: str | nn.Module | None,
    preprocessing: ReIDPreprocessing | None,
) -> LoadPlan:
    """Resolve a ``from_pretrained`` request into the values it needs.

    Resolution follows three load forms (in order):

    1. **Default curated alias** — both ``source`` and ``architecture`` omitted;
       ``DEFAULT_MODEL`` is used.
    2. **Curated alias or config directory** — ``source`` resolves via
       :func:`reid.catalog.resolve_model_card` (alias, ``ReIDModel.save``
       directory, HF repo, or ``http(s)://`` folder URL). When ``architecture`` is explicitly overridden,
       the override wins (with a warning); the card's weights and preprocessing
       are retained.
    3. **Bare weights file** — ``source`` is a local path or ``hf://`` / ``gd://`` / ``http(s)://``
       URL that does not resolve to a model card; ``architecture`` is required.

    Returns:
        A ``LoadPlan`` with architecture, weights, preprocessing, an optional
        domain warning, and the share of weight tensors the file must match.
        Callers build the architecture, load ``weights`` when present, and enforce
        ``required_match_fraction`` when set.

    Raises:
        ValueError: For an ``architecture`` without a ``source`` (random weights come from
            constructing ``ReIDModel(architecture)``), a bare weights file without ``architecture``, or a local
            directory that is missing ``reid_config.json``.
    """
    from reid.catalog import DEFAULT_MODEL, resolve_model_card

    if source is None and architecture is None:
        source = DEFAULT_MODEL

    resolved = resolve_model_card(source) if source is not None else None
    if resolved is not None:
        card, config_preprocessing = resolved
        arch_overridden = False
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
        required_fraction = _MISMATCH_FRACTION if arch_overridden else 1.0
        return LoadPlan(
            resolved_arch,
            card.weights,
            _resolve_preprocessing(preprocessing, config_preprocessing, card.architecture),
            card.domain_warning,
            required_fraction,
        )

    if source is None:
        raise ValueError(
            "from_pretrained loads trained weights, so it needs a source when an architecture is given. "
            f"For a randomly initialised model, use ReIDModel({architecture!r})."
        )

    if architecture is None:
        raise ValueError(
            f"Cannot load {source!r}: it appears to be a bare weights "
            "file but no `architecture` was provided. Pass e.g. "
            "architecture='osnet_x1_0' to specify the model "
            "architecture, or point to a directory produced by "
            "ReIDModel.save() for a self-describing checkpoint."
        )
    return LoadPlan(
        architecture,
        source,
        _resolve_preprocessing(preprocessing, None, architecture),
        None,
        _MISMATCH_FRACTION,
    )


# ---------------------------------------------------------------------------
# State-dict loading
# ---------------------------------------------------------------------------


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


def _read_state_dict(path: str, device: torch.device, *, allow_pickle: bool = True) -> dict:
    """Read a raw state dict from a ``.pth`` or ``.safetensors`` file."""
    if path.endswith(".safetensors"):
        return load_file(path, device=str(device))

    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    except (TypeError, pickle.UnpicklingError, RuntimeError) as exc:
        if not allow_pickle:
            raise ValueError(
                f"{path!r} only loads with weights_only=False, which runs arbitrary code, so reid does not use it "
                "for a file downloaded from a URL. Convert the checkpoint to .safetensors, or download it and "
                "pass the local path if you trust its source."
            ) from exc
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


def load_state_dict(
    module: nn.Module,
    path: str,
    device: torch.device,
    *,
    architecture: str | None = None,
    drop_prefixes: tuple[str, ...] = _DEFAULT_DROP_PREFIXES,
    remap: Callable[[dict], dict] | None = None,
    warn_threshold: float = _MISMATCH_FRACTION,
    required_match_fraction: float | None = None,
    label: str | None = None,
    allow_pickle: bool = True,
) -> None:
    """Load ``path`` into ``module`` by name and shape (classifier keys skipped).

    The matched fraction and the missing/unexpected key lists are logged. A
    partial load (missing or unexpected keys) emits a ``UserWarning``. When
    ``required_match_fraction`` is set and the matched fraction falls short, a
    ``ValueError`` is raised. Nothing is returned.

    Args:
        module: Target module to load weights into (updated in place).
        path: Checkpoint file (``.pth`` or ``.safetensors``).
        device: Device to map the checkpoint onto.
        architecture: When set, resolve checkpoint key remapping via
            :func:`reid.architectures.checkpoint_remap_for_architecture`.
            Ignored when ``remap`` is passed explicitly.
        drop_prefixes: Key prefixes to skip (e.g. classifier heads). Used only
            when neither ``architecture`` nor ``remap`` supplies a remapper.
        remap: Optional key remapper applied instead of the default prefix rules.
            When set, takes precedence over ``architecture``.
        warn_threshold: Below this matched fraction the warning treats the load
            as a likely architecture mismatch rather than an expected partial.
        required_match_fraction: If set, raise when the matched fraction is below
            this value.
        label: Optional human-readable name used in the unmet-
            ``required_match_fraction`` error message. Defaults to
            ``architecture`` when omitted.
        allow_pickle: Whether a ``.pth`` that fails the safe
            ``weights_only=True`` load may fall back to ``weights_only=False``.
            ``False`` raises ``ValueError`` instead.
    """
    if remap is None and architecture:
        from reid.architectures import checkpoint_remap_for_architecture

        remap = checkpoint_remap_for_architecture(architecture)

    if label is None:
        label = architecture or None

    state_dict = _read_state_dict(path, device, allow_pickle=allow_pickle)

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
    total = len(target)
    matched_fraction = len(matched) / total if total else 0.0
    summary = (
        f"loaded {len(matched)}/{total} params ({matched_fraction:.0%}); "
        f"{len(missing)} missing, {len(unexpected)} unused"
    )

    logger.info(
        "State-dict load from %r: %s; missing=%s; unexpected=%s",
        path,
        summary,
        missing,
        unexpected,
    )

    if missing or unexpected:
        if matched_fraction < warn_threshold:
            warnings.warn(
                f"Only {summary} from {path!r}. The weights likely do not "
                f"match this architecture — check the model/weights pairing.",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"Partial load from {path!r}: {summary}. Missing "
                "parameters keep their initialised values; unused checkpoint "
                "keys were ignored.",
                UserWarning,
                stacklevel=2,
            )

    if required_match_fraction is not None and matched_fraction < required_match_fraction:
        target_name = f"{label!r} " if label else ""
        raise ValueError(
            f"Checkpoint {path!r} matched only {len(matched)} of {total} "
            f"{target_name}weight tensors (required >= {required_match_fraction:.0%}). "
            f"{summary}"
        )
