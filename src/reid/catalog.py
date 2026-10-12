# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Curated ReID aliases, ``ModelCard``, and ``reid_config.json`` I/O."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.parse import urlparse

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

from reid.architectures.fastreid_sbs import FASTREID_SBS_ARCHITECTURE
from reid.loaders import resolve_weights
from reid.preprocessing import ReIDPreprocessing

_WEIGHTS_FILE_SUFFIXES = (".safetensors", ".pth", ".pt")

# OSNet x1.0 trained on MSMT17 with combineall (train + query + gallery).
# Strong general-purpose pedestrian features, but combineall includes MSMT17
# test IDs — not suitable for MSMT17 evaluation.
_DEFAULT_OSNET_WEIGHTS = (
    "hf://kaiyangzhou/osnet/"
    "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10"
    "_softmax_labelsmooth_flip_jitter.pth"
)

_DOMAIN_WARNING = (
    "The default ReIDModel weights were trained on MSMT17 pedestrian images. "
    "Performance may degrade significantly on non-person objects (vehicles, "
    "animals, products, etc.). Pass a domain-specific checkpoint via "
    "`ReIDModel.from_pretrained(source, architecture=...)` for other use cases."
)

# ---------------------------------------------------------------------------
# Public constants and types
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "osnet_x1_0_msmt17_combineall"
"""Curated alias used when ``ReIDModel.from_pretrained`` gets no ``source``."""

# BoT-SORT MOT17 SBS-S50 (ResNeSt50 + GeM + BNNeck), trained on MOT17 train-half GT crops.
# Weights: https://github.com/niraharon/bot-sort#model-zoo
_FASTREID_MOT17_WEIGHTS = "gd://1QZFWpoa80rqo7O-HXmlss8J8CnS7IUsN/mot17_sbs_S50.pth"
_FASTREID_MOT17_WARNING = (
    "The fastreid_mot17_sbs50 weights were trained on MOT17 pedestrian crops. "
    "Use this encoder for MOT tracking benchmarks (BoT-SORT-ReID replication); "
    "cross-domain retrieval on Market-1501 / MSMT17 is expected to underperform."
)

FASTREID_MOT17_SBS50 = "fastreid_mot17_sbs50"


@dataclass
class ModelCard:
    """Architecture, weights source, and optional domain metadata.

    Produced by curated ``ALIASES`` entries or by loading ``reid_config.json``
    from a ``save_pretrained`` directory / HF repo. Preprocessing for curated
    aliases is derived from the architecture at load time; saved configs may
    carry a preprocessing override in ``reid_config.json`` (see
    :func:`resolve_model_card`).
    """

    architecture: str
    weights: str | None
    domain_warning: str | None = None


ALIASES: dict[str, ModelCard] = {
    DEFAULT_MODEL: ModelCard(
        architecture="osnet_x1_0",
        weights=_DEFAULT_OSNET_WEIGHTS,
        domain_warning=_DOMAIN_WARNING,
    ),
    FASTREID_MOT17_SBS50: ModelCard(
        architecture=FASTREID_SBS_ARCHITECTURE,
        weights=_FASTREID_MOT17_WEIGHTS,
        domain_warning=_FASTREID_MOT17_WARNING,
    ),
}

# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def list_aliases() -> list[str]:
    """Return curated alias names (sorted keys of :data:`ALIASES`)."""
    return sorted(ALIASES)


def resolve_model_card(source: str) -> tuple[ModelCard, ReIDPreprocessing | None] | None:
    """Resolve an alias or a ``save_pretrained`` model to a load recipe.

    A saved model is a folder holding ``reid_config.json`` and ``weights.safetensors``:
    a local directory, an ``hf://org/repo`` repository, or an ``http(s)://`` folder URL.
    Curated aliases return ``(card, None)``; saved models return ``(card, preprocessing)``
    with the preprocessing from ``reid_config.json``. Returns ``None`` when ``source`` is
    a bare weights file, which needs an explicit architecture.

    Raises:
        ValueError: If a local directory or ``http(s)://`` folder URL has no
            ``reid_config.json`` (folders are never bare weight files).
        RuntimeError: If a Hugging Face repository's ``reid_config.json`` cannot be
            downloaded.
    """
    if source in ALIASES:
        return ALIASES[source], None
    saved_model = _find_saved_model(source)
    if saved_model is None:
        return None
    config_path, weights_source = saved_model
    return _parse_config_file(config_path, weights_source=weights_source)


def save_model_config(
    card: ModelCard,
    directory: str,
    *,
    preprocessing: ReIDPreprocessing,
) -> None:
    """Write ``reid_config.json`` to ``directory``."""
    if not isinstance(card.architecture, str):
        raise ValueError(
            "Cannot save a ModelCard whose architecture is an nn.Module. "
            "Use a named architecture string (e.g. 'osnet_x1_0', "
            "'timm:resnet50') to enable save_pretrained()."
        )

    config = {
        "architecture": card.architecture,
        "preprocessing": preprocessing.to_dict(),
    }
    config_path = os.path.join(directory, "reid_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_saved_model(source: str) -> tuple[str, str] | None:
    """Locate the two files of the ``save_pretrained`` folder that ``source`` names.

    ``source`` may be a local directory, an ``hf://org/repo`` repository, or an
    ``http(s)://`` folder URL. The config is read right away to build the model card,
    so a remote one is downloaded here; the weights are only located, and are
    downloaded later, when the model loads them.

    Returns:
        ``(config_path, weights_source)``: the local path of ``reid_config.json``, and
        the path or URL of ``weights.safetensors``. ``None`` when ``source`` is not a
        folder, such as a bare weights file.

    Raises:
        ValueError: If a local directory or ``http(s)://`` folder URL has no
            ``reid_config.json``.
        RuntimeError: If a Hugging Face repository's ``reid_config.json`` cannot be
            downloaded.
    """
    if os.path.isdir(source):
        config_path = os.path.join(source, "reid_config.json")
        if not os.path.isfile(config_path):
            raise ValueError(
                f"Cannot load {source!r}: directory has no reid_config.json. "
                "Point to a directory produced by ReIDModel.save_pretrained(), "
                "or pass a weights *file* path together with architecture=... ."
            )
        return config_path, os.path.join(source, "weights.safetensors")

    # hf://org/repo has exactly 2 parts after "hf://"; hf://org/repo/file.pth is a bare weights file.
    if source.startswith("hf://") and len([p for p in source[len("hf://") :].split("/") if p]) == 2:
        try:
            config_path = _download_hf_config(source)
        except (HfHubHTTPError, EntryNotFoundError, OSError, ValueError) as exc:
            raise RuntimeError(f"Failed to resolve Hugging Face repo config for {source!r}.") from exc
        return config_path, source.rstrip("/") + "/weights.safetensors"

    # An http(s) URL is a folder unless it ends in a weights file name.
    if source.startswith(("http://", "https://")) and not urlparse(source).path.endswith(_WEIGHTS_FILE_SUFFIXES):
        folder_url = source.rstrip("/")
        return _download_url_config(folder_url), f"{folder_url}/weights.safetensors"

    return None


def _download_hf_config(hf_repo_url: str) -> str:
    """Download ``reid_config.json`` from an HF repo URL and return its local path."""
    parts = [p for p in hf_repo_url[len("hf://") :].split("/") if p]
    repo_id = "/".join(parts[:2])
    revision = None
    if "@" in repo_id:
        repo_id, revision = repo_id.split("@", 1)
    return hf_hub_download(repo_id=repo_id, filename="reid_config.json", revision=revision)


def _download_url_config(folder_url: str) -> str:
    """Download ``reid_config.json`` from a folder served over http(s) and return its local path."""
    config_url = f"{folder_url}/reid_config.json"
    try:
        return resolve_weights(config_url)
    except RuntimeError as exc:
        raise ValueError(
            f"Cannot load {folder_url!r}: no reid_config.json at {config_url}. "
            "Point to a folder produced by ReIDModel.save_pretrained(), "
            "or pass a weights *file* URL together with architecture=... ."
        ) from exc


def _parse_config_file(
    config_path: str,
    *,
    weights_source: str,
) -> tuple[ModelCard, ReIDPreprocessing]:
    """Parse a ``reid_config.json`` file into a ``ModelCard`` and preprocessing."""
    with open(config_path) as f:
        data = json.load(f)

    architecture = data["architecture"]
    preprocessing = ReIDPreprocessing.from_dict(data.get("preprocessing", {}))
    return (
        ModelCard(
            architecture=architecture,
            weights=weights_source,
        ),
        preprocessing,
    )
