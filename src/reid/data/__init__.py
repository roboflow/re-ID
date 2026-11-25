# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Shared ReID dataset types, benchmark loaders, and crop partitioners."""

from __future__ import annotations

from reid.data.benchmarks import (
    load_market1501,
    load_market1501_test,
    load_market1501_train,
    load_msmt17,
    load_msmt17_test,
    load_msmt17_train,
    load_msmt17_val,
)
from reid.data.identity_crops import load_identity_folders
from reid.data.splits import ReIDSplit

__all__ = [
    "ReIDSplit",
    "load_identity_folders",
    "load_market1501",
    "load_market1501_test",
    "load_market1501_train",
    "load_msmt17",
    "load_msmt17_test",
    "load_msmt17_train",
    "load_msmt17_val",
]
