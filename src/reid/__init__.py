# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import importlib

__all__ = [
    "DEFAULT_MODEL",
    "FASTREID_MOT17_SBS50",
    "KeyReport",
    "ModelCard",
    "ReIDEncoder",
    "ReIDEvaluator",
    "ReIDMetrics",
    "ReIDModel",
    "ReIDPreprocessing",
    "ReIDResult",
    "ReIDSplit",
    "build_architecture",
    "checkpoint_remap_for_architecture",
    "compute_reid_metrics",
    "default_preprocessing_for_architecture",
    "list_architectures",
    "load_market1501",
    "load_model_config",
    "load_msmt17",
    "load_state_dict_for_architecture",
    "load_state_dict_into",
    "resolve_model_card",
    "resolve_weights",
    "save_model_config",
]

REID_INSTALL_HINT = (
    "ReID features require optional dependencies. Install with: pip install 'roboflow-reid'"
)

_REID_OPTIONAL_ROOTS = frozenset(
    {
        "torch",
        "torchvision",
        "timm",
        "huggingface_hub",
        "safetensors",
        "PIL",
        "gdown",
    }
)

# NumPy-only symbols — safe to import without torch/timm/HF.
from reid.architectures import build_architecture, checkpoint_remap_for_architecture, list_architectures
from reid.encoder import ReIDEncoder
from reid.eval import (
    ReIDEvaluator,
    ReIDMetrics,
    ReIDResult,
    ReIDSplit,
    compute_reid_metrics,
    load_market1501,
    load_msmt17,
)
from reid.models import (
    DEFAULT_MODEL,
    FASTREID_MOT17_SBS50,
    KeyReport,
    ModelCard,
    ReIDPreprocessing,
    default_preprocessing_for_architecture,
    load_model_config,
    load_state_dict_for_architecture,
    load_state_dict_into,
    resolve_model_card,
    resolve_weights,
    save_model_config,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ReIDModel": ("reid.model", "ReIDModel"),
}


def _is_optional_reid_import_error(exc: ImportError) -> bool:
    name = getattr(exc, "name", None)
    if name is None:
        return False
    root = name.split(".", 1)[0]
    return root in _REID_OPTIONAL_ROOTS


def _import_reid_symbol(module_name: str, attr_name: str) -> object:
    """Import a heavy ReID symbol, rewriting missing-extra errors only."""
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    except ImportError as exc:
        if _is_optional_reid_import_error(exc):
            raise ImportError(REID_INSTALL_HINT) from exc
        raise


def __getattr__(name: str) -> object:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = _import_reid_symbol(module_name, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
