# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Trainable wrapper for ``timm`` backbones."""

from __future__ import annotations

import timm
import torch
from torch import nn

from reid.architectures.encoder import TrainableEncoder


class TimmEncoder(TrainableEncoder):
    """A ``timm`` trunk with the re-ID training head.

    The trunk is created with ``num_classes=0`` so it emits pooled embeddings;
    an identity classifier is attached for :func:`reid.training.train_reid`.
    Weights save under ``trunk.*`` keys; inference loads strip that prefix, so
    a saved checkpoint reloads into the plain ``timm`` model built by
    :func:`reid.architectures.build_architecture` at ``num_classes=0``.
    """

    _head_modules = frozenset({"classifier"})

    def __init__(self, model_name: str, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        self.trunk = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feature_dim = int(self.trunk.num_features)
        if num_classes > 0:
            self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return trunk embeddings (mode-independent)."""
        return self.trunk(x)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


def remap_state_dict_onto_timm_encoder(state_dict: dict) -> dict:
    """Map a plain ``timm`` or reid-saved checkpoint onto :class:`TimmEncoder`.

    Only re-prefixes trunk keys with ``trunk.`` (idempotently). Nothing is
    dropped by name: :func:`reid.loaders.load_state_dict` already ignores keys
    the target lacks, which filters classifier heads without also discarding
    trunk parameters that live under a head-like name (ConvNeXt keeps its final
    ``head.norm`` LayerNorm there even at ``num_classes=0``).
    """
    return {f"trunk.{key.removeprefix('trunk.')}": value for key, value in state_dict.items()}
