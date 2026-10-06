# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Shared ReID dataset types, benchmark loaders, and crop partitioners."""

from __future__ import annotations

from reid.data.benchmarks import load_market1501, load_msmt17
from reid.data.identity_crops import ReIDDataset
from reid.data.splits import ReIDSplit

__all__ = [
    "ReIDDataset",
    "ReIDSplit",
    "load_market1501",
    "load_msmt17",
]
