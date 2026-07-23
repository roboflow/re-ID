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

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

from reid.architectures.fastreid_sbs import FASTREID_SBS_ARCHITECTURE
from reid.preprocessing import ReIDPreprocessing

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
    """Resolve an alias or ``reid_config.json`` directory/repo to a load recipe.

    Returns ``None`` when ``source`` is a bare weights *file* URL/path without a
    config. Curated aliases return ``(card, None)``; saved / HF configs return
    ``(card, preprocessing)`` with the preprocessing dict from
    ``reid_config.json``.

    Raises:
        ValueError: If ``source`` is a local directory that does not contain
            ``reid_config.json`` (directories are never bare weight files).
    """
    if source in ALIASES:
        return ALIASES[source], None

    # Local directory must be a save_pretrained() tree (reid_config.json).
    if os.path.isdir(source):
        config_path = os.path.join(source, "reid_config.json")
        if not os.path.isfile(config_path):
            raise ValueError(
                f"Cannot load {source!r}: directory has no reid_config.json. "
                "Point to a directory produced by ReIDModel.save_pretrained(), "
                "or pass a weights *file* path together with architecture=... ."
            )
        return load_model_config(source)

    # HF repo URL (no trailing filename) — try to fetch reid_config.json.
    # hf://org/repo has exactly 2 non-empty parts after "hf://".
    # hf://org/repo/file.pth has 3+ parts and is a bare weights file.
    if source.startswith("hf://"):
        rest = source[len("hf://") :]
        parts = [p for p in rest.split("/") if p]
        if len(parts) == 2:
            try:
                return _load_hf_repo_config(source)
            except (HfHubHTTPError, EntryNotFoundError, OSError, ValueError) as exc:
                raise RuntimeError(f"Failed to resolve Hugging Face repo config for {source!r}.") from exc

    return None


def load_model_config(directory_or_repo: str) -> tuple[ModelCard, ReIDPreprocessing]:
    """Load a ``ModelCard`` and preprocessing from ``reid_config.json``.

    Local directories may also include ``weights.safetensors``.
    """
    if directory_or_repo.startswith("hf://"):
        card, preprocessing = _load_hf_repo_config(directory_or_repo)
        if preprocessing is None:
            raise ValueError(f"No preprocessing found in config for {directory_or_repo!r}.")
        return card, preprocessing

    config_path = os.path.join(directory_or_repo, "reid_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No reid_config.json found in {directory_or_repo!r}. "
            "Use ReIDModel.save_pretrained() to create a self-describing "
            "model directory."
        )
    weights_path = os.path.join(directory_or_repo, "weights.safetensors")
    weights_source = weights_path if os.path.exists(weights_path) else None
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


def _load_hf_repo_config(hf_repo_url: str) -> tuple[ModelCard, ReIDPreprocessing | None]:
    """Download and parse ``reid_config.json`` from an HF repo URL."""
    rest = hf_repo_url[len("hf://") :]
    parts = [p for p in rest.split("/") if p]
    repo_id = "/".join(parts[:2])
    revision = None
    if "@" in repo_id:
        repo_id, revision = repo_id.split("@", 1)

    config_path = hf_hub_download(
        repo_id=repo_id,
        filename="reid_config.json",
        revision=revision,
    )
    weights_source = hf_repo_url.rstrip("/") + "/weights.safetensors"
    return _parse_config_file(config_path, weights_source=weights_source)


def _parse_config_file(
    config_path: str,
    *,
    weights_source: str | None,
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
