# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Shared ReID dataset types, benchmark loaders, and crop partitioners."""

from __future__ import annotations

from reid.data.benchmarks import load_market1501, load_msmt17
from reid.data.identity_crops import (
    build_identity_crop_holdout_split,
    build_identity_crop_retrieval_split,
    identity_name_from_crop_path,
    load_identity_crops,
)
from reid.data.splits import MARKET1501_GALLERY_JUNK_PIDS, ReIDSplit

__all__ = [
    "MARKET1501_GALLERY_JUNK_PIDS",
    "ReIDSplit",
    "build_identity_crop_holdout_split",
    "build_identity_crop_retrieval_split",
    "identity_name_from_crop_path",
    "load_identity_crops",
    "load_market1501",
    "load_msmt17",
]
