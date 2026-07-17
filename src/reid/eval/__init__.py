# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from reid.eval.datasets import ReIDSplit, load_market1501, load_msmt17
from reid.eval.evaluator import ReIDEvaluator, ReIDResult
from reid.eval.metrics import ReIDMetrics, compute_reid_metrics

__all__ = [
    "ReIDEvaluator",
    "ReIDMetrics",
    "ReIDResult",
    "ReIDSplit",
    "compute_reid_metrics",
    "load_market1501",
    "load_msmt17",
]
