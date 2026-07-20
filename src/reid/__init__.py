# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""re-ID: appearance re-identification models and gallery evaluation.

The public surface is intentionally narrow. Advanced helpers live in
submodules: checkpoint loading in :mod:`reid.loaders`, the curated alias
catalog and ``ModelCard`` in :mod:`reid.catalog`, and architecture dispatch
in :mod:`reid.architectures`.
"""

from __future__ import annotations

from reid.architectures import list_architectures
from reid.catalog import DEFAULT_MODEL, FASTREID_MOT17_SBS50, list_aliases
from reid.eval import (
    ReIDEvaluator,
    ReIDMetrics,
    ReIDResult,
    ReIDSplit,
    compute_reid_metrics,
    load_market1501,
    load_msmt17,
)
from reid.model import ReIDModel
from reid.preprocessing import ReIDPreprocessing

__all__ = [
    "DEFAULT_MODEL",
    "FASTREID_MOT17_SBS50",
    "ReIDEvaluator",
    "ReIDMetrics",
    "ReIDModel",
    "ReIDPreprocessing",
    "ReIDResult",
    "ReIDSplit",
    "compute_reid_metrics",
    "list_aliases",
    "list_architectures",
    "load_market1501",
    "load_msmt17",
]
