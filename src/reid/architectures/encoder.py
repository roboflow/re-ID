# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Shared training head for re-ID encoder architectures."""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import nn


class TrainableEncoder(nn.Module):
    """Base class implementing the :class:`~reid.training.TrainableBackbone` contract.

    Subclasses set ``feature_dim``, build ``self.classifier`` when
    ``num_classes > 0``, implement :meth:`_embed`, and declare
    ``_head_modules``: the top-level module names kept trainable (and left in
    ``train()`` mode) while the trunk is frozen.
    """

    _head_modules: ClassVar[frozenset[str]]
    feature_dim: int

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return embeddings regardless of train/eval mode."""
        raise NotImplementedError

    def forward_embeddings_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(embeddings, logits)`` for CE + triplet training."""
        if not hasattr(self, "classifier"):
            raise TypeError(
                f"{type(self).__name__} was built without a classifier head "
                "(num_classes=0). Pass num_classes>0 for training."
            )
        embeddings = self._embed(x)
        return embeddings, self.classifier(embeddings)

    def set_backbone_frozen(self, frozen: bool) -> None:
        """Freeze/unfreeze trunk params; keep ``_head_modules`` trainable.

        When frozen, trunk modules are switched to ``eval()`` so BatchNorm
        running statistics stop updating. Unfreezing restores trunk ``train()``.
        """
        for name, param in self.named_parameters():
            top = name.split(".", 1)[0]
            param.requires_grad = (top in self._head_modules) or (not frozen)

        for name, module in self.named_children():
            if name in self._head_modules:
                continue
            module.train(not frozen)
