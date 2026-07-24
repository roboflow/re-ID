# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
#
# Adapted from JDAI-CV/fast-reid (Apache-2.0)
# Copyright 2019 JD.com Inc. JD AI
# Source: https://github.com/JDAI-CV/fast-reid
#   - GeM: fastreid/layers/pooling.py (GeneralizedMeanPooling / GeneralizedMeanPoolingP)
#   - SBS head layout: fastreid/modeling/heads/embedding_head.py
#   - last_stride=1 ResNeSt semantics: fastreid/modeling/backbones/resnest.py
# Also used via BoT-SORT's FastReIDInterface (NirAharon/BoT-SORT, MIT).
# ------------------------------------------------------------------------

"""FastReID Strong Baseline (SBS) inference stack for BoT-SORT checkpoints.

Uses ``timm`` ``resnest50d`` with a small FastReID ``last_stride=1`` patch on
``layer4[0]``, then Generalized Mean Pooling and a BatchNorm neck — the same
inference path as FastReID ``EmbeddingHead`` in eval.
"""

from __future__ import annotations

from typing import Any, cast

import timm
import torch
from torch import nn
from torch.nn import functional as F

from reid.preprocessing import ReIDPreprocessing

FASTREID_SBS_ARCHITECTURE = "fastreid_sbs_resnest50"
FASTREID_SBS_EMBED_DIM = 2048


def _patch_resnest50d_for_fastreid_last_stride(backbone: nn.Module) -> nn.Module:
    """Align ``timm`` ``resnest50d`` (``output_stride=16``) with FastReID ResNeSt.

    FastReID ``LAST_STRIDE=1`` keeps layer4 spatial stride at 1 via:
    - downsample avg-pool kernel 1 (identity spatial size), and
    - average-downsampling (AVD) after the Split-Attention conv with stride 1.

    ``timm``'s dilated / stride-16 path instead uses ``AvgPool2dSame(2, stride=1)``
    and leaves ``avd_last`` unset on ``layer4[0]``, which shifts embeddings even
    when checkpoint keys load 100%.
    """
    backbone_any = cast(Any, backbone)
    block = backbone_any.layer4[0]
    block.downsample[0] = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
    block.avd_last = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    return backbone


class GeneralizedMeanPooling(nn.Module):
    """GeM pooling used by FastReID SBS (learnable exponent *p*).

    ``p`` defaults to 3.0 only until checkpoint load; MOT17 SBS-S50 stores the
    trained value under ``heads.pool_layer.p`` (≈1.72). See
    :func:`remap_fastreid_sbs_state_dict`.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.pow(1.0 / self.p)


class FastReIDSBSResNeSt50(nn.Module):
    """ResNeSt50 SBS re-ID encoder matching BoT-SORT / FastReID MOT17 checkpoints.

    Inference path: patched ``timm`` ResNeSt-50 → GeM (:attr:`pool`) → BNNeck
    (:attr:`bottleneck`). Weights for all three stages load from a BoT-SORT
    ``.pth`` via :func:`remap_fastreid_sbs_state_dict`.

    When ``num_classes > 0``, a linear identity classifier is attached for
    :func:`reid.training.train_reid` (same layout as FastReID ``EmbeddingHead``;
    classifier keys are dropped again on inference load).
    """

    @classmethod
    def default_preprocessing(cls) -> ReIDPreprocessing:
        """Default crop/tensor recipe for FastReID SBS (BoT-SORT FastReIDInterface).

        Config (aligned with BoT-SORT / FastReID MOT17 SBS inference):
        - input size: 384x128 (H, W)
        - resize: stretch (cv2.resize, no letterbox)
        - colour: BGR crops converted to RGB
        - normalize: ImageNet mean/std
        """
        return ReIDPreprocessing(
            input_size=(384, 128),
            resize_mode="stretch",
            to_rgb=True,
        )

    def __init__(self, num_classes: int = 0) -> None:
        super().__init__()
        self.feature_dim = FASTREID_SBS_EMBED_DIM
        backbone = timm.create_model(
            "resnest50d",
            pretrained=False,
            num_classes=0,
            global_pool="",
            output_stride=16,
        )
        self.backbone = _patch_resnest50d_for_fastreid_last_stride(backbone)
        self.pool = GeneralizedMeanPooling()
        self.bottleneck = nn.BatchNorm1d(FASTREID_SBS_EMBED_DIM)
        if num_classes > 0:
            self.classifier = nn.Linear(FASTREID_SBS_EMBED_DIM, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return BNNeck embeddings (inference and training-time feature path)."""
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        return self.bottleneck(x)

    def forward_embeddings_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(embeddings, logits)`` for CE + triplet training."""
        if not hasattr(self, "classifier"):
            raise TypeError(
                "FastReIDSBSResNeSt50 was built without a classifier head "
                "(num_classes=0). Pass num_classes>0 for training."
            )
        embeddings = self.forward(x)
        return embeddings, self.classifier(embeddings)

    def set_backbone_frozen(self, frozen: bool) -> None:
        """Freeze/unfreeze the ResNeSt trunk; keep pool / BNNeck / classifier trainable.

        When frozen, the trunk is switched to ``eval()`` so BatchNorm running
        statistics stop updating. Unfreezing restores trunk ``train()``.
        """
        for param in self.backbone.parameters():
            param.requires_grad = not frozen
        self.backbone.train(not frozen)


def build_fastreid_sbs_resnest50(*, num_classes: int = 0, pretrained: bool = False) -> FastReIDSBSResNeSt50:
    """Build the BoT-SORT FastReID SBS ResNeSt50 encoder (weights loaded separately)."""
    del pretrained
    return FastReIDSBSResNeSt50(num_classes=num_classes)


def remap_fastreid_sbs_state_dict(state_dict: dict) -> dict:
    """Map BoT-SORT / FastReID SBS checkpoint keys onto :class:`FastReIDSBSResNeSt50`.

    Renames FastReID ``heads.*`` keys to ``pool.*`` / ``bottleneck.*`` and passes
    ``backbone.*`` through unchanged. Native ``save_pretrained`` keys
    (``pool.*``, ``bottleneck.*``) pass through as-is. Skips classifier weights
    (``heads.weight`` / ``classifier.*``); unused at inference.
    """
    mapped: dict = {}
    for key, value in state_dict.items():
        key = key[7:] if key.startswith("module.") else key
        if key.startswith(("backbone.", "pool.", "bottleneck.")):
            mapped[key] = value
        elif key == "heads.pool_layer.p":
            mapped["pool.p"] = value
        elif key.startswith("heads.bottleneck.0."):
            mapped["bottleneck." + key[len("heads.bottleneck.0.") :]] = value
        # Skip heads.weight / classifier.* (identity classifier; unused at inference).
    return mapped
