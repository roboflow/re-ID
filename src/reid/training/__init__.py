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
:class:`~reid.training.losses.ReIDLoss` from their defining modules.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reid.training.trainer import (
        TrainConfig,
        TrainResult,
        train_reid,
    )

__all__ = [
    "TrainConfig",
    "TrainResult",
    "train_reid",
]

_LAZY_TRAINER_SYMBOLS = frozenset({"TrainConfig", "TrainResult", "train_reid"})


def __getattr__(name: str):
    if name in _LAZY_TRAINER_SYMBOLS:
        from reid.training import trainer

        value = getattr(trainer, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
