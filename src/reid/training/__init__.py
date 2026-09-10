# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""ReID fine-tuning via :func:`train_reid`.

Crop loading and query/gallery partitioning live in :mod:`reid.data.identity_crops`;
MOT crop generation lives in :mod:`reid.data.mot`. Custom loops can import
:class:`~reid.training.datasets.ReIDTrainDataset`,
:class:`~reid.training.datasets.PKSampler`, and
:class:`~reid.training.losses.ReIDLoss` from their defining modules; third-party
backbones implement the :class:`TrainableBackbone` protocol.
"""

from __future__ import annotations

from reid.training.trainer import TrainableBackbone, TrainConfig, TrainResult, train_reid

__all__ = [
    "TrainableBackbone",
    "TrainConfig",
    "TrainResult",
    "train_reid",
]
