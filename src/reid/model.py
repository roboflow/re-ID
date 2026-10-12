# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""ReID appearance encoder: loading, inference, and checkpoint I/O.

Array-type boundary: torch tensors stay inside the model; embeddings cross to
``numpy`` once, on the way out of :meth:`ReIDModel._embed_batch`, because
``supervision`` and ``trackers`` consume numpy. Everything downstream
(``reid.eval``) is numpy throughout.
"""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Sequence

import numpy as np
import supervision as sv
import torch
import torch.nn as nn
from PIL import Image as PILImage
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from reid.architectures import build_architecture
from reid.catalog import ModelCard, save_model_config
from reid.checks import validate_positive_int
from reid.crop_dataset import CropDataset
from reid.data import ReIDSplit
from reid.eval import EvaluationProtocol, ReIDResult, RetrievalProtocol
from reid.eval.metrics import _compute_reid_metrics, _distance_matrix
from reid.loaders import load_state_dict, resolve_load_plan, resolve_weights
from reid.preprocessing import ReIDPreprocessing

logger = logging.getLogger(__name__)

# Distances reach up to 4x the largest squared embedding length, which must stay within float32.
_MAX_SQUARED_LENGTH = float(np.finfo(np.float32).max) / 4


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


def _best_device() -> torch.device:
    """Return the best available compute device (``cuda`` → ``mps`` → ``cpu``)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _select_device(device: str) -> torch.device:
    """Resolve ``"auto"`` or a device string to a ``torch.device``."""
    if device == "auto":
        return _best_device()
    return torch.device(device)


class ReIDModel:
    """Appearance encoder with loading, preprocessing, and checkpoint I/O.

    Provides ``extract_embeddings`` (for video-frame crops) and
    ``extract_embeddings_from_paths`` (for on-disk crops), plus
    ``from_pretrained`` and ``save_pretrained``. The default checkpoint is
    pedestrian-trained; pass ``source`` / ``architecture`` for other domains.

    Args:
        backbone: Encoder module (``(B, 3, H, W)`` → ``(B, D)`` in eval).
        device: Device the backbone runs on.
        preprocessing: Crop and embedding preprocessing.
        architecture: Registered architecture name, required by
            ``save_pretrained``. ``None`` for a raw ``nn.Module`` backbone.
    """

    def __init__(
        self,
        backbone: nn.Module,
        device: torch.device,
        preprocessing: ReIDPreprocessing,
        architecture: str | None = None,
    ) -> None:
        self._backbone = backbone
        # Encoders are inference-only; force eval() so callers that pass a raw
        # nn.Module (e.g. still in train mode) get deterministic BN/dropout.
        self._backbone.eval()
        self._device = device
        self._preprocessing = preprocessing
        self._crop_transform = preprocessing.build_crop_transform()
        self._architecture = architecture
        logger.info("ReIDModel preprocessing: %s", preprocessing.describe())

    @property
    def device(self) -> torch.device:
        """Device the backbone runs on."""
        return self._device

    def to(self, device: str | torch.device) -> ReIDModel:
        """Move the encoder to ``device`` and return self, for chaining."""
        target = _select_device(device) if isinstance(device, str) else device
        self._backbone.to(target)
        self._device = target
        return self

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

        ``source`` may be a curated alias, ``hf://`` repo or file, an
        ``http(s)://`` file or folder URL, a local path or directory with
        ``reid_config.json``, or ``None`` for the default model. A bare
        ``.pth`` / ``.safetensors`` file requires ``architecture``.

        Args:
            source: Model source (alias, Hub URL, file or folder URL, local path, or ``None``).
            architecture: Backbone override; required for bare weight files, and
                must be the architecture the weights were trained with.
            preprocessing: Preprocessing override.
            device: Compute device (``"auto"`` picks the best available).

        Returns:
            Loaded model with the backbone on ``device``.

        Raises:
            ValueError: If a bare weights file is given without ``architecture``,
                or the weights cover less of the architecture than required:
                all of it for a pretrained name or saved model, half for a bare
                weights file.
        """
        resolved_device = _select_device(device)

        # Resolve the load recipe (architecture, weights, preprocessing, domain
        # warning, and strictness) in one place.
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
            load_state_dict(
                backbone,
                local_path,
                resolved_device,
                architecture=arch_name or None,
                required_match_fraction=plan.required_match_fraction,
            )

        backbone.to(resolved_device)

        return cls(
            backbone,
            resolved_device,
            plan.preprocessing,
            architecture=plan.architecture if isinstance(plan.architecture, str) else None,
        )

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
        state_dict = {key: value.detach().cpu() for key, value in self._backbone.state_dict().items()}
        save_file(state_dict, weights_path)

        card = ModelCard(
            architecture=self._architecture,
            weights=None,
        )
        save_model_config(card, directory, preprocessing=self._preprocessing)
        logger.info("ReIDModel saved to %s", directory)

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        data: ReIDSplit,
        gallery: ReIDSplit | None = None,
        *,
        protocol: EvaluationProtocol | None = None,
        batch_size: int = 64,
        return_distmat: bool = False,
        verbose: bool = True,
    ) -> ReIDResult:
        """Evaluate this model under an evaluation protocol.

        The protocol selects which images are compared and which metrics are
        reported; this model embeds them. Selection is deterministic, so the
        same data and protocol score identical inputs for every model.

        Args:
            data: Images to evaluate. The protocol splits them into query and
                gallery images; when ``gallery`` is provided, they are the query
                images.
            gallery: Gallery images for the queries in ``data``. When provided,
                both splits are used as given; re-association builds its own
                gallery and rejects one.
            protocol: Evaluation protocol. Defaults to
                :class:`~reid.eval.RetrievalProtocol`.
            batch_size: Images per embedding batch.
            return_distmat: Keep the query-by-gallery distance matrix in the result.
            verbose: Log progress.

        Returns:
            :class:`~reid.eval.ReIDResult` with protocol metrics as percentages,
            the number of scored queries, and protocol metadata.

        Raises:
            TypeError: If ``protocol`` is not an ``EvaluationProtocol``.
            ValueError: If an input is invalid, the model returns non-finite
                embeddings or embeddings too large to compare, or no query can
                be scored.
        """
        validate_positive_int(batch_size, "batch_size")
        if protocol is None:
            protocol = RetrievalProtocol()
        elif not isinstance(protocol, EvaluationProtocol):
            raise TypeError("protocol must be an EvaluationProtocol instance.")
        query, gallery_split, candidate_mask, metadata = protocol.build_query_gallery(data, gallery)

        # A crop may be both a query and a gallery image. Embed it once.
        paths = list(dict.fromkeys([*query.image_paths, *gallery_split.image_paths]))
        if verbose:
            logger.info("Extracting embeddings (%s images)…", len(paths))
        embeddings = self.extract_embeddings_from_paths(paths, batch_size=batch_size)
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("The model returned non-finite embeddings.")
        row = {path: index for index, path in enumerate(paths)}
        query_embeddings = embeddings[[row[path] for path in query.image_paths]]
        gallery_embeddings = embeddings[[row[path] for path in gallery_split.image_paths]]

        distmat = _distance_matrix(query_embeddings, gallery_embeddings, protocol.distance)
        metrics = _compute_reid_metrics(
            distmat=distmat,
            q_pids=query.pids,
            g_pids=gallery_split.pids,
            q_camids=query.camids,
            g_camids=gallery_split.camids,
            gallery_junk_pids=gallery_split.gallery_junk_pids,
            candidate_mask=candidate_mask,
            exclude_same_pid_camid=protocol.exclude_same_pid_camid,
        )
        if metrics.num_queries == 0:
            raise ValueError(
                "No query has a correct match among its candidates, so nothing can be scored. "
                "Check that the query identities appear in the gallery."
            )
        if verbose:
            logger.info("Results (%s)\n%s", protocol.distance, metrics)
        skipped = metadata.get("num_skipped_queries", 0) + len(query) - metrics.num_queries
        return ReIDResult(
            metrics=protocol.metric_values(metrics),
            num_queries=metrics.num_queries,
            distmat=distmat if return_distmat else np.empty((0, 0), dtype=np.float32),
            metadata={
                **metadata,
                "exclude_same_pid_camid": protocol.exclude_same_pid_camid,
                "num_skipped_queries": skipped,
            },
        )

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
        if out.size > 0 and np.square(out, dtype=np.float64).sum(axis=1).max() >= _MAX_SQUARED_LENGTH:
            raise ValueError(
                f"embeddings are too large to compare in float32: each squared length must stay below "
                f"{_MAX_SQUARED_LENGTH:.3g}"
            )
        return out

    def extract_embeddings_from_paths(
        self,
        image_paths: Sequence[str],
        *,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> np.ndarray:
        """Extract embeddings from pre-cropped image paths.

        For bbox crops from a video frame, use ``extract_embeddings``. Images are
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
            ValueError: If ``batch_size`` is not a positive integer, or the
                model returns non-finite embeddings or embeddings too large to
                compare.
        """
        if not image_paths:
            return np.empty((0, 0), dtype=np.float32)
        validate_positive_int(batch_size, "batch_size")

        dataset = CropDataset(image_paths, self._preprocessing.build_crop_transform())
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=torch.stack,
        )

        all_embeddings: list[np.ndarray] = [self._embed_batch(batch) for batch in loader]
        return np.concatenate(all_embeddings, axis=0)

    def extract_embeddings(
        self,
        detections: sv.Detections,
        frame: np.ndarray,
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Extract appearance embeddings for each detection.

        Crops are sub-batched so crowded frames do not build one oversized
        forward pass. Shares crop→tensor→forward with
        ``extract_embeddings_from_paths``. Raw (un-normalised) embeddings are
        returned; cosine callers normalise at distance time.

        Args:
            detections: Detections whose ``xyxy`` boxes define the crops.
            frame: BGR video frame the detections were produced on.
            batch_size: Crops per forward pass.

        Returns:
            Float32 array of shape ``(N, D)``, or ``(0, 0)`` when empty.

        Raises:
            ValueError: If ``batch_size`` is not a positive integer, or the
                model returns non-finite embeddings or embeddings too large to
                compare.
        """
        if len(detections) == 0:
            return np.empty((0, 0), dtype=np.float32)
        validate_positive_int(batch_size, "batch_size")

        frame_h, frame_w = frame.shape[:2]
        tensors: list[torch.Tensor] = []
        for box in detections.xyxy:
            safe_box = _clamp_xyxy_to_frame(box, frame_h, frame_w)
            crop = sv.crop_image(image=frame, xyxy=safe_box.astype(int))
            crop = self._preprocessing.prepare_crop_array(crop, from_bgr_frame=True)
            tensors.append(self._crop_transform(PILImage.fromarray(crop)))

        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(tensors), batch_size):
            chunk = tensors[start : start + batch_size]
            all_embeddings.append(self._embed_batch(torch.stack(chunk)))
        return np.concatenate(all_embeddings, axis=0)
