# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Evaluation protocols, results, and retrieval metrics."""

from __future__ import annotations

from reid.eval.metrics import ReIDMetrics, ReIDResult
from reid.eval.protocols import EvaluationProtocol, ReassociationProtocol, RetrievalProtocol

__all__ = [
    "EvaluationProtocol",
    "ReIDMetrics",
    "ReIDResult",
    "ReassociationProtocol",
    "RetrievalProtocol",
]
