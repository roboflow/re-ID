# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""ReID appearance encoder: loading, inference, and checkpoint I/O."""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Callable, Sequence

import numpy as np
import supervision as sv
import torch
import torch.nn as nn
from PIL import Image as PILImage
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset

from reid.architectures import build_architecture
from reid.catalog import ModelCard, save_model_config
from reid.device import best_device
from reid.loaders import load_state_dict_for_architecture, resolve_load_plan, resolve_weights
from reid.preprocessing import ReIDPreprocessing

logger = logging.getLogger(__name__)


class _CropPathDataset(Dataset):
    """Reads crop images from disk and applies preprocessing on worker processes.

    Images on disk are assumed RGB-ready, so the ``to_rgb`` BGR→RGB swap (which
    is for video frames) is intentionally not applied here.
    """

    def __init__(
        self,
        image_paths: Sequence[str],
        preprocessing: ReIDPreprocessing,
        transforms: Callable[[np.ndarray], torch.Tensor],
    ) -> None:
        self._image_paths = list(image_paths)
        self._preprocessing = preprocessing
        self._transforms = transforms

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        img = PILImage.open(self._image_paths[index]).convert("RGB")
        resized = self._preprocessing.resize_crop(np.asarray(img))
        return self._transforms(resized)


def _clamp_xyxy_to_frame(box: np.ndarray, height: int, width: int) -> np.ndarray:
    """Clip a box to frame bounds with at least 1 px width and height."""
    x1, y1, x2, y2 = box.astype(float)
    max_x = max(width, 1)
    max_y = max(height, 1)
    x1 = float(np.clip(x1, 0, max_x - 1))
    y1 = float(np.clip(y1, 0, max_y - 1))
    x2 = float(np.clip(x2, x1 + 1, max_x))
    y2 = float(np.clip(y2, y1 + 1, max_y))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _select_device(device: str) -> torch.device:
    """Resolve ``"auto"`` or a device string to a ``torch.device``."""
    if device == "auto":
        return best_device()
    return torch.device(device)


class ReIDModel:
    """Appearance encoder with loading, preprocessing, and checkpoint I/O.

    Provides ``extract_features`` (for video-frame crops) and
    ``extract_features_from_paths`` (for on-disk crops), plus
    ``from_pretrained`` and ``save_pretrained``. The default checkpoint is
    pedestrian-trained; pass ``source`` / ``architecture`` for other domains.

    Args:
        backbone: Feature-extractor module (``(B, 3, H, W)`` → ``(B, D)`` in eval).
        device: Device the backbone runs on.
        preprocessing: Crop and embedding preprocessing.
    """

    def __init__(
        self,
        backbone: nn.Module,
        device: torch.device,
        preprocessing: ReIDPreprocessing,
    ) -> None:
        self._backbone = backbone
        # Encoders are inference-only; force eval() so callers that pass a raw
        # nn.Module (e.g. still in train mode) get deterministic BN/dropout.
        self._backbone.eval()
        self._device = device
        self._preprocessing = preprocessing
        self._transforms = preprocessing.build_transform()
        # Set by from_pretrained() to enable save_pretrained(); None if the
        # model was constructed directly with a raw nn.Module architecture.
        self._architecture: str | None = None
        logger.info("ReIDModel preprocessing: %s", preprocessing.describe())

    @property
    def preprocessing(self) -> ReIDPreprocessing:
        """Active preprocessing pipeline."""
        return self._preprocessing

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pretrained(
        cls,
        source: str | None = None,
        *,
        architecture: str | nn.Module | None = None,
        preprocessing: ReIDPreprocessing | None = None,
        device: str = "auto",
    ) -> ReIDModel:
        """Load a ``ReIDModel`` from an alias, Hub/local checkpoint, or architecture-only init.

        ``source`` may be a curated alias, ``hf://`` repo or file, a local path
        or directory with ``reid_config.json``, or ``None`` for the default
        model. A bare ``.pth`` / ``.safetensors`` file requires ``architecture``.

        Args:
            source: Model source (alias, Hub URL, local path, or ``None``).
            architecture: Backbone override; required for bare weight files.
            preprocessing: Preprocessing override.
            device: Compute device (``"auto"`` picks the best available).

        Returns:
            Loaded model with the backbone on ``device``.

        Raises:
            ValueError: If a bare weights file is given without ``architecture``.
        """
        resolved_device = _select_device(device)

        # Resolve the full load recipe (architecture, weights, preprocessing,
        # domain warning, and strictness) in one place.
        plan = resolve_load_plan(source, architecture, preprocessing)

        if plan.domain_warning:
            warnings.warn(plan.domain_warning, UserWarning, stacklevel=2)

        # Always build offline here. Checkpoint weights (when present) are loaded
        # explicitly below; architecture-only builds stay randomly initialised.
        backbone = build_architecture(plan.architecture, pretrained=False)
        backbone.eval()

        if plan.weights is not None:
            local_path = resolve_weights(plan.weights)
            arch_name = plan.architecture if isinstance(plan.architecture, str) else ""
            logger.info("ReIDModel loading weights (%s)", plan.weights)
            load_state_dict_for_architecture(
                backbone,
                local_path,
                resolved_device,
                arch_name,
                required_match_fraction=plan.required_match_fraction,
            )

        backbone.to(resolved_device)

        instance = cls(backbone, resolved_device, plan.preprocessing)
        # Store the architecture name (string only) to enable save_pretrained.
        instance._architecture = plan.architecture if isinstance(plan.architecture, str) else None
        return instance

    def save_pretrained(self, directory: str) -> None:
        """Persist ``weights.safetensors`` and ``reid_config.json`` for later loads.

        Args:
            directory: Output directory (created if missing).

        Raises:
            ValueError: If the architecture name is unknown.
        """
        if self._architecture is None:
            raise ValueError(
                "Cannot save a model whose architecture name is unknown. "
                "Build the model via from_pretrained() with a named "
                "architecture (e.g. 'osnet_x1_0', 'timm:resnet50') to "
                "enable save_pretrained()."
            )

        os.makedirs(directory, exist_ok=True)

        weights_path = os.path.join(directory, "weights.safetensors")
        save_file(self._backbone.state_dict(), weights_path)

        card = ModelCard(
            architecture=self._architecture,
            weights=None,
            preprocessing=self._preprocessing,
        )
        save_model_config(card, directory)
        logger.info("ReIDModel saved to %s", directory)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def _crop_to_tensor(self, rgb_crop: np.ndarray) -> torch.Tensor:
        """Resize an RGB crop to the model input and apply tensor normalisation."""
        resized = self._preprocessing.resize_crop(rgb_crop)
        return self._transforms(resized)

    def _embed_batch(self, batch: torch.Tensor) -> np.ndarray:
        """Run one pre-collated crop batch through the backbone → ``(B, D)`` float32."""
        batch = batch.to(self._device)
        with torch.inference_mode():
            embeddings = self._backbone(batch)
        out = embeddings.cpu().numpy().astype(np.float32)
        if out.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {out.shape}")
        if out.size > 0 and not np.all(np.isfinite(out)):
            raise ValueError("embeddings must contain only finite values")
        return out

    def extract_features_from_paths(
        self,
        image_paths: Sequence[str],
        *,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> np.ndarray:
        """Extract embeddings from pre-cropped image paths.

        For bbox crops from a video frame, use ``extract_features``. Images are
        decoded and preprocessed via a ``DataLoader`` so ``num_workers > 0``
        parallelises disk I/O on platforms that support it. Raw (un-normalised)
        embeddings are returned; cosine callers normalise at distance time.

        Args:
            image_paths: Paths to RGB-ready crop images.
            batch_size: Images per forward pass.
            num_workers: DataLoader worker processes (``0`` decodes inline).

        Returns:
            Float32 array of shape ``(N, D)``.

        Raises:
            ValueError: If ``batch_size`` is less than 1.
        """
        if not image_paths:
            return np.empty((0, 0), dtype=np.float32)
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        dataset = _CropPathDataset(image_paths, self._preprocessing, self._transforms)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=torch.stack,
        )

        all_embeddings: list[np.ndarray] = [self._embed_batch(batch) for batch in loader]
        return np.concatenate(all_embeddings, axis=0)

    def extract_features(
        self,
        detections: sv.Detections,
        frame: np.ndarray,
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Extract appearance embeddings for each detection.

        Crops are sub-batched so crowded frames do not build one oversized
        forward pass. Shares crop→tensor→forward with
        ``extract_features_from_paths``. Raw (un-normalised) embeddings are
        returned; cosine callers normalise at distance time.

        Args:
            detections: Detections whose ``xyxy`` boxes define the crops.
            frame: BGR video frame the detections were produced on.
            batch_size: Crops per forward pass.

        Returns:
            Float32 array of shape ``(N, D)``, or ``(0, 0)`` when empty.

        Raises:
            ValueError: If ``batch_size`` is less than 1.
        """
        if len(detections) == 0:
            return np.empty((0, 0), dtype=np.float32)
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        frame_h, frame_w = frame.shape[:2]
        tensors: list[torch.Tensor] = []
        for box in detections.xyxy:
            safe_box = _clamp_xyxy_to_frame(box, frame_h, frame_w)
            crop = sv.crop_image(image=frame, xyxy=safe_box.astype(int))
            if self._preprocessing.to_rgb:
                crop = crop[:, :, ::-1].copy()
            tensors.append(self._crop_to_tensor(crop))

        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(tensors), batch_size):
            chunk = tensors[start : start + batch_size]
            all_embeddings.append(self._embed_batch(torch.stack(chunk)))
        return np.concatenate(all_embeddings, axis=0)
