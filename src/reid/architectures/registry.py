# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Architecture construction, listing, remapping, and preprocessing defaults."""

from __future__ import annotations

from collections.abc import Callable

import timm
import torch.nn as nn

from reid.architectures.fastreid_sbs import (
    FASTREID_SBS_ARCHITECTURE,
    FastReIDSBSResNeSt50,
    build_fastreid_sbs_resnest50,
    remap_fastreid_sbs_state_dict,
)
from reid.architectures.osnet import OSNet, build_osnet
from reid.architectures.timm_encoder import TimmEncoder
from reid.preprocessing import ReIDPreprocessing

_OSNET_VARIANTS: dict[str, str] = {
    "osnet_x0_25": "x0_25",
    "osnet_x0_5": "x0_5",
    "osnet_x0_75": "x0_75",
    "osnet_x1_0": "x1_0",
}

_DEFAULT_PREPROCESSING_BY_ARCH: dict[str, Callable[[], ReIDPreprocessing]] = {
    **{name: OSNet.default_preprocessing for name in _OSNET_VARIANTS},
    FASTREID_SBS_ARCHITECTURE: FastReIDSBSResNeSt50.default_preprocessing,
}

_TIMM_PREFIX = "timm:"


def build_architecture(
    architecture: str | nn.Module,
    *,
    num_classes: int = 0,
    pretrained: bool = False,
) -> nn.Module:
    """Build a backbone from ``osnet_*``, ``timm:<name>``, or a pre-built module.

    ``pretrained=True`` only affects ``timm:<name>`` models (ImageNet weights).
    OSNet ignores it because checkpoints are always loaded explicitly.

    Args:
        architecture: Registered name, ``timm:<name>``, or an ``nn.Module``.
        num_classes: Classifier head size (``0`` for embedding-only).
        pretrained: Load ImageNet weights for ``timm:`` models when ``True``.

    Returns:
        Backbone module ready for weight loading or inference.
    """
    if not isinstance(architecture, str):
        return architecture

    if architecture.startswith(_TIMM_PREFIX):
        name = architecture[len(_TIMM_PREFIX) :]
        if num_classes > 0:
            return TimmEncoder(name, num_classes=num_classes, pretrained=pretrained)
        return timm.create_model(name, pretrained=pretrained, num_classes=0)

    if architecture in _OSNET_VARIANTS:
        return build_osnet(variant=_OSNET_VARIANTS[architecture], num_classes=num_classes)

    if architecture == FASTREID_SBS_ARCHITECTURE:
        return build_fastreid_sbs_resnest50(
            num_classes=num_classes,
            pretrained=pretrained,
        )

    raise ValueError(
        f"Unknown architecture {architecture!r}. Choose a registered name "
        f"({list_architectures()}), a timm model as 'timm:<name>' (e.g. "
        f"'timm:resnet50'), or pass a torch.nn.Module instance."
    )


def list_architectures() -> list[str]:
    """Return registered architecture names (timm models use ``timm:<name>``)."""
    return sorted([*_OSNET_VARIANTS, FASTREID_SBS_ARCHITECTURE])


def checkpoint_remap_for_architecture(
    architecture: str,
) -> Callable[[dict], dict] | None:
    """Return the registered checkpoint key remapper, if any."""
    if architecture == FASTREID_SBS_ARCHITECTURE:
        return remap_fastreid_sbs_state_dict
    return None


def default_preprocessing_for_architecture(
    architecture: str | None,
) -> ReIDPreprocessing:
    """Return the preprocessing defaults owned by ``architecture``."""
    if architecture is None:
        return OSNet.default_preprocessing()
    if architecture in _DEFAULT_PREPROCESSING_BY_ARCH:
        return _DEFAULT_PREPROCESSING_BY_ARCH[architecture]()
    return OSNet.default_preprocessing()
