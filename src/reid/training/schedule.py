# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""The learning-rate schedule of ``ReIDModel.train``, a warmup followed by the default drops or the user's scheduler."""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    CyclicLR,
    LambdaLR,
    LinearLR,
    LRScheduler,
    OneCycleLR,
    ReduceLROnPlateau,
    SequentialLR,
)

# Schedulers that train, which steps the scheduler once per epoch with no score, can't run
SCHEDULERS_TRAIN_CANNOT_STEP = (ReduceLROnPlateau, OneCycleLR, CyclicLR, ChainedScheduler, SequentialLR)


def build_lr_schedule_with_warmup(
    optimizer: Optimizer,
    epochs: int,
    warmup_epochs: int,
    lr_scheduler: type[LRScheduler] | None,
    lr_scheduler_kwargs: dict[str, object] | None,
) -> LRScheduler:
    """Return the learning-rate schedule ``ReIDModel.train`` steps once per epoch.

    Epochs count from 1. Warmup epoch ``i`` trains at ``lr * i / warmup_epochs``, so the last warmup epoch reaches
    ``lr``. After the warmup:

    - Without ``lr_scheduler``, training stays at ``lr`` until epoch ``epochs * 3 // 4``, then trains at ``lr / 10``
      until epoch ``epochs * 7 // 8``, then at ``lr / 100``. With the defaults of 40 epochs and 5 of warmup, epochs 1
      to 5 climb from ``lr / 5`` to ``lr``, 6 to 30 train at ``lr``, 31 to 35 at ``lr / 10``, and 36 to 40 at
      ``lr / 100``. On short runs, the first drop comes after epoch 1 at the earliest, and the second at least one
      epoch after the first and after the warmup, so the run never takes both at once.
    - With ``lr_scheduler``, ``lr_scheduler(optimizer, **lr_scheduler_kwargs)`` takes over from ``lr`` at the first
      epoch after the warmup, and is stepped once per epoch.

    Args:
        optimizer: The run's optimizer, holding ``lr``.
        epochs: Epochs in the run.
        warmup_epochs: Epochs of linear warmup, and 0 skips it.
        lr_scheduler: Scheduler class that takes over after the warmup, or ``None`` for the default drops.
        lr_scheduler_kwargs: Arguments for ``lr_scheduler``.
    """
    if lr_scheduler is None:
        # on short runs, keep the first drop after epoch 1, and the second after the warmup and after the first drop
        first_drop = max(1, epochs * 3 // 4)
        second_drop = max(first_drop + 1, warmup_epochs + 1, epochs * 7 // 8)
        # LambdaLR counts epochs from 0
        return LambdaLR(optimizer, lambda index: _lr_factor(index + 1, warmup_epochs, (first_drop, second_drop)))
    after_warmup = lr_scheduler(optimizer, **(lr_scheduler_kwargs or {}))
    if warmup_epochs == 0:
        return after_warmup
    warmup = LinearLR(optimizer, start_factor=1 / warmup_epochs, total_iters=warmup_epochs - 1)
    return SequentialLR(optimizer, [warmup, after_warmup], milestones=[warmup_epochs])


def _lr_factor(epoch: int, warmup_epochs: int, drops: tuple[int, int]) -> float:
    """Return the multiplier of ``lr`` for ``epoch``, counted from 1, under the default schedule."""
    # During the warmup, climb linearly to lr
    if epoch <= warmup_epochs:
        return epoch / warmup_epochs
    # After it, a tenth for each drop already passed
    drops_passed = [drop for drop in drops if drop < epoch]
    return 0.1 ** len(drops_passed)
