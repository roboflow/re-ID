# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""re-ID: appearance re-identification models and gallery evaluation.

The public surface is intentionally narrow. Advanced helpers live in
submodules: checkpoint loading in :mod:`reid.loaders`, the curated alias
catalog and ``ModelCard`` in :mod:`reid.catalog`, shared dataset types and
benchmark loaders in :mod:`reid.data`, and architecture dispatch in
:mod:`reid.architectures`.
"""

from __future__ import annotations

from reid.architectures import list_architectures
from reid.catalog import DEFAULT_MODEL, FASTREID_MOT17_SBS50, list_aliases
from reid.data import ReIDSplit, load_identity_folders, load_market1501, load_msmt17
from reid.eval import (
    ReassociationProtocol,
    ReIDResult,
    RetrievalProtocol,
)
from reid.model import ReIDModel
from reid.preprocessing import Interpolation, ReIDPreprocessing

__all__ = [
    "DEFAULT_MODEL",
    "FASTREID_MOT17_SBS50",
    "Interpolation",
    "ReassociationProtocol",
    "ReIDModel",
    "ReIDPreprocessing",
    "ReIDResult",
    "ReIDSplit",
    "RetrievalProtocol",
    "list_aliases",
    "list_architectures",
    "load_identity_folders",
    "load_market1501",
    "load_msmt17",
]
