# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Building blocks of ``ReIDModel.train``: losses, PK batches, augmentation, the schedule, one epoch, and its outputs.

Only ``TripletLoss`` is public; the rest is internal.
"""

from __future__ import annotations

from reid.training.losses import TripletLoss

__all__ = ["TripletLoss"]
