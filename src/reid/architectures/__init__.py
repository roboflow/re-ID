# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""ReID architecture public API."""

from __future__ import annotations

from reid.architectures.registry import (
    build_architecture,
    checkpoint_remap_for_architecture,
    default_preprocessing_for_architecture,
    list_architectures,
)

__all__ = [
    "build_architecture",
    "checkpoint_remap_for_architecture",
    "default_preprocessing_for_architecture",
    "list_architectures",
]
