# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""ReID appearance encoder: loading, training, inference, and checkpoint I/O.

Array-type boundary: torch tensors stay inside the model; embeddings cross to
``numpy`` once, on the way out of :meth:`ReIDModel._embed_batch`, because
``supervision`` and ``trackers`` consume numpy. Everything downstream
(``reid.eval``) is numpy throughout.
"""

from __future__ import annotations

import logging
import math
import os
import random
import tempfile
import time
import warnings
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path

import numpy as np
import supervision as sv
import torch
import torch.nn as nn
from PIL import Image as PILImage
from safetensors.torch import save_file
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, StackDataset
from torchvision.transforms import ColorJitter, RandomErasing, RandomHorizontalFlip

from reid.architectures import build_architecture, default_preprocessing_for_architecture
from reid.catalog import ModelCard, save_model_config
from reid.checks import validate_int_at_least, validate_positive_int
from reid.crop_dataset import CropDataset
from reid.data import ReIDSplit
from reid.eval import EvaluationProtocol, ReassociationProtocol, ReIDResult, RetrievalProtocol
from reid.eval.metrics import _compute_reid_metrics, distance_matrix
from reid.loaders import load_state_dict, resolve_load_plan, resolve_weights
from reid.preprocessing import ReIDPreprocessing
from reid.training.epoch import LOSS_NAMES, train_one_epoch
from reid.training.losses import TripletLoss
from reid.training.outputs import append_metrics_row, format_epoch_line, write_training_config
from reid.training.sampling import PKSampler, build_sequence_groups
from reid.training.schedule import SCHEDULERS_TRAIN_CANNOT_STEP, build_lr_schedule_with_warmup
from reid.training.transforms import build_train_transform

logger = logging.getLogger(__name__)


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
    """Appearance encoder with loading, training, preprocessing, and checkpoint I/O.

    Provides ``extract_embeddings`` (for video-frame crops) and
    ``extract_embeddings_from_paths`` (for on-disk crops), plus
    ``from_pretrained``, ``save`` and ``train``.

    Constructing it builds the architecture with randomly initialised weights, without downloading anything. Its
    embeddings carry no identity information until it is trained; to load trained weights, use
    :meth:`from_pretrained`.

    Args:
        architecture: Registered architecture name (see ``list_architectures``), ``timm:<name>``, or a pre-built
            ``nn.Module`` (``(B, 3, H, W)`` → ``(B, D)``, in eval mode and, to train it, in train mode).
        preprocessing: Crop and embedding preprocessing. Defaults to the architecture's own preprocessing.
        device: Compute device (``"auto"`` picks the best available).
    """

    def __init__(
        self,
        architecture: str | nn.Module,
        *,
        preprocessing: ReIDPreprocessing | None = None,
        device: str | torch.device = "auto",
    ) -> None:
        self._architecture = architecture if isinstance(architecture, str) else None
        self._backbone = build_architecture(architecture, pretrained=False)
        self._backbone.eval()
        self.to(device)
        if preprocessing is None:
            preprocessing = default_preprocessing_for_architecture(self._architecture)
        self._preprocessing = preprocessing
        self._crop_transform = preprocessing.build_crop_transform()
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

    @cached_property
    def embedding_dim(self) -> int:
        """Length of the embeddings this model returns, measured once by embedding a dummy crop."""
        height, width = self._preprocessing.input_size
        dummy_crop = torch.zeros(1, 3, height, width, device=self._device)  # only the output's shape matters
        with torch.inference_mode():
            dummy_embedding = self._backbone(dummy_crop)
        return dummy_embedding.shape[1]

    @property
    def _architecture_name(self) -> str:
        """The architecture's name, or the module's class name, which ``save`` and ``training_config.json`` record."""
        return self._architecture if self._architecture is not None else type(self._backbone).__name__

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pretrained(
        cls,
        source: str,
        *,
        architecture: str | nn.Module | None = None,
        preprocessing: ReIDPreprocessing | None = None,
        device: str = "auto",
    ) -> ReIDModel:
        """Load a ``ReIDModel`` with trained weights from an alias or a Hub, web or local checkpoint.

        ``source`` may be a curated alias, ``hf://`` repo or file, an ``http(s)://`` file or folder URL, or a local path
        or directory with ``reid_config.json``. A bare ``.pth`` / ``.safetensors`` file requires ``architecture``.

        Args:
            source: Model source (alias, Hub URL, file or folder URL, or local path).
            architecture: Backbone override; required for bare weight files, and
                must be the architecture the weights were trained with.
            preprocessing: Preprocessing override.
            device: Compute device (``"auto"`` picks the best available).

        Returns:
            Loaded model with the backbone on ``device``.

        Raises:
            ValueError: If a bare weights file is given without ``architecture``, or the weights cover less of the
                architecture than required: all of it for a pretrained name or saved model, half for a bare weights
                file.
        """
        # Resolve the load recipe (architecture, weights, preprocessing, domain
        # warning, and strictness) in one place.
        plan = resolve_load_plan(source, architecture, preprocessing)

        if plan.domain_warning:
            warnings.warn(plan.domain_warning, UserWarning, stacklevel=2)

        model = cls(plan.architecture, preprocessing=plan.preprocessing, device=device)

        if plan.weights is not None:
            local_path = resolve_weights(plan.weights)
            logger.info("ReIDModel loading weights (%s)", plan.weights)
            load_state_dict(
                model._backbone,
                local_path,
                model.device,
                architecture=model._architecture,
                required_match_fraction=plan.required_match_fraction,
                allow_pickle=not plan.weights.startswith(("http://", "https://")),
            )

        return model

    def save(self, directory: str, verbose: bool = True) -> None:
        """Save the model's current weights and its configuration, so ``from_pretrained(directory)`` loads it back.

        Writes the weights to ``weights.safetensors`` and the architecture and preprocessing to ``reid_config.json``.
        A model built from a module records the module's class name, such as ``"MyNet"``, and loads back with
        ``from_pretrained(directory, architecture=MyNet())``. Each file is written to a temporary folder first and then
        moved into ``directory``, so a crash during the save leaves each file whole, from this save or the one before.

        Args:
            directory: Output directory (created if missing).
            verbose: Log the folder the model was saved to, at ``INFO`` level.
        """
        os.makedirs(directory, exist_ok=True)

        # Write the files into a temporary folder, then move each one into directory in a single step
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            state_dict = {key: value.detach().cpu() for key, value in self._backbone.state_dict().items()}
            save_file(state_dict, os.path.join(temporary, "weights.safetensors"))
            card = ModelCard(
                architecture=self._architecture_name,
                weights=None,
            )
            save_model_config(card, temporary, preprocessing=self._preprocessing)
            for name in os.listdir(temporary):
                os.replace(os.path.join(temporary, name), os.path.join(directory, name))
        if verbose:
            logger.info("ReIDModel saved to %s", directory)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(
        self,
        train_data: ReIDSplit,
        output_dir: str | Path | None = None,
        val_data: ReIDSplit | None = None,
        val_gallery: ReIDSplit | None = None,
        val_protocol: EvaluationProtocol | None = None,
        checkpoint_metric: str | None = "auto",
        epochs: int = 40,
        steps_per_epoch: int | None = None,
        identities_per_batch: int = 8,
        crops_per_identity: int = 4,
        sample_per_sequence: bool = False,
        lr: float = 3e-4,
        warmup_epochs: int = 5,
        weight_decay: float = 5e-4,
        optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict[str, object] | None = None,
        lr_scheduler: type[LRScheduler] | None = None,
        lr_scheduler_kwargs: dict[str, object] | None = None,
        freeze_backbone_epochs: int = 3,
        classification_loss: nn.Module = nn.CrossEntropyLoss(label_smoothing=0.1),
        embedding_loss: nn.Module = TripletLoss(margin=0.3),
        classification_loss_weight: float = 1.0,
        embedding_loss_weight: float = 1.0,
        augmentations: Sequence[nn.Module] = (
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.2, contrast=0.15, saturation=0.15),
            RandomErasing(p=0.5, value="random"),
        ),
        num_workers: int = 0,
        mixed_precision: bool = True,
        seed: int = 42,
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        """Fine-tune this model on identity crops in place on its ``device``, and return each epoch's losses and scores.

        Training adds a classification layer over the training identities and minimises the total loss
        ``classification_loss_weight * classification_loss + embedding_loss_weight * embedding_loss``, where
        ``classification_loss`` scores the classification layer's outputs and ``embedding_loss`` the embeddings. The
        learning rate warms up linearly to ``lr``, then follows the ``lr_scheduler`` passed. Without one, it drops to a
        tenth after epoch ``epochs * 3 // 4`` and again after epoch ``epochs * 7 // 8``. A drop that falls inside the
        warmup moves to its end, and the second drop always comes at least one epoch after the first.

        Start from trained weights loaded with ``from_pretrained``. ``ReIDModel(architecture)`` starts from random
        weights. The model ends in eval mode with the last epoch's weights and every parameter trainable, and
        ``save`` writes it without the classification head.

        With ``val_data``, we score the model with ``evaluate`` before training and after every epoch. With
        ``output_dir`` too, ``best/`` holds the epoch with the highest ``checkpoint_metric``, a tie going to the later
        epoch. The score before training is never picked. If no epoch beats it, training made the model worse on
        ``val_data``, and a warning says so. Validation doesn't change training's random draws, so a seed trains the
        same way with or without it.

        With ``output_dir``, ``train`` writes ``training_config.json`` before the first epoch, then after every epoch
        saves the model to ``last/``, as ``save`` does, and appends the epoch to ``metrics.csv``. With ``val_data``
        and a ``checkpoint_metric``, each epoch that becomes the best is saved to ``best/``.

        Args:
            train_data: Training crops and their person IDs, for example from ``load_identity_folders``.
            output_dir: Folder for ``last/``, ``best/``, ``metrics.csv`` and ``training_config.json``. ``None`` writes
                nothing.
            val_data: Validation crops and their person IDs, scored before training and after every epoch. With
                ``val_gallery``, the query crops and their person IDs. ``None`` trains without validation.
            val_gallery: Gallery crops and their person IDs, for a validation set that comes with its own query and
                gallery. ``None`` lets ``val_protocol`` split ``val_data`` into both.
            val_protocol: How validation is scored, a ``RetrievalProtocol`` or a ``ReassociationProtocol`` as in
                ``evaluate``. ``None`` means ``RetrievalProtocol()``.
            checkpoint_metric: Metric that picks the epoch saved to ``best/``, by the name the protocol reports it
                under, such as ``"rank1"``. ``"auto"`` means ``"map"`` for ``RetrievalProtocol`` and ``"mrr"`` for
                ``ReassociationProtocol``. ``None`` scores every epoch and writes no ``best/``.
            epochs: Number of epochs.
            steps_per_epoch: Batches per epoch. ``None`` makes an epoch as many batches as the crops fill,
                ``len(train_data) // (identities_per_batch * crops_per_identity)`` and at least 1.
            identities_per_batch: Identities in a batch, at least 2.
            crops_per_identity: Crops of each identity in a batch, at least 2.
            sample_per_sequence: Build every batch from one sequence, for tracking crops where a person ID only
                holds inside its sequence. The sequence is read from each crop's path,
                ``{sequence}_{track}/{sequence}_{frame}.jpg`` as ``generate_mot_patches`` names it. A sequence with
                fewer than ``identities_per_batch`` identities gives smaller batches, so an epoch sees fewer crops
                than one pass. A sequence with a single identity is left out, with a warning. ``False`` mixes
                sequences.
            lr: Learning rate after the warmup.
            warmup_epochs: Epochs of linear warmup, where epoch ``i`` trains at ``lr * i / warmup_epochs``.
                ``0`` starts at ``lr``.
            weight_decay: The optimizer's weight decay.
            optimizer: Optimizer class, built as ``optimizer(parameters, lr=lr, weight_decay=weight_decay,
                **optimizer_kwargs)``.
            optimizer_kwargs: Any other arguments the ``optimizer`` class takes, as listed in its ``torch.optim``
                documentation, such as ``{"momentum": 0.9}`` for ``SGD``.
            lr_scheduler: Scheduler class that takes over after the warmup, built as
                ``lr_scheduler(optimizer, **lr_scheduler_kwargs)`` and stepped once per epoch. ``None`` uses the
                drops above. ``ReduceLROnPlateau``, ``OneCycleLR``, ``CyclicLR``, ``ChainedScheduler`` and
                ``SequentialLR`` raise, since ``train`` steps the scheduler once per epoch with no score.
            lr_scheduler_kwargs: The arguments ``lr_scheduler`` takes besides the optimizer, as listed in its
                ``torch.optim.lr_scheduler`` documentation, such as ``{"T_max": epochs - warmup_epochs}`` for
                ``CosineAnnealingLR``.
            freeze_backbone_epochs: Epochs at the start that train only the classification layer and the
                backbone's modules named in its ``trainable_when_frozen`` (OSNet's ``fc``,
                FastReID SBS's ``pool`` and ``bottleneck``). A backbone without it trains only the classification
                layer. The other modules keep their BatchNorm statistics. Warns if this covers every epoch.
            classification_loss: Loss on the classification layer's scores, called as ``loss(scores, labels)``.
            embedding_loss: Loss on the embeddings, called as ``loss(embeddings, labels)``, such as
                ``TripletLoss(margin=0.5)``.
            classification_loss_weight: Weight of ``classification_loss`` in the total. ``0`` trains without it.
            embedding_loss_weight: Weight of ``embedding_loss`` in the total. ``0`` trains without it.
            augmentations: ``torchvision.transforms`` objects, run in order on the resized crop. ``RandomErasing``
                runs after normalisation, since it works on tensors. ``()`` turns augmentation off.
            num_workers: Processes that load and augment crops. ``0`` does it in the main process.
            mixed_precision: Train in mixed precision, running most operations in float16 instead of float32, which
                is faster and uses less memory. It runs only on CUDA and has no effect on other devices.
            seed: Seed for Python's ``random``, numpy and torch. On CPU, the same call on the same weights repeats the
                run.
            verbose: Show a progress bar with the epoch's mean loss so far, and print one line after each epoch. With
                ``val_data`` and a ``checkpoint_metric``, also print the best epoch and its score at the end.

        Returns:
            One dict per epoch with ``epoch``, ``lr`` (the learning rate the epoch trained at), ``train/loss`` (the
            weighted total), ``train/classification_loss`` and ``train/embedding_loss`` (both unweighted), the
            validation metrics with ``val_data``, such as ``val/map`` and ``val/rank1``, and ``time`` (seconds,
            validation included). With ``val_data``, the list starts with the scores before training, a dict with
            ``epoch`` 0 that holds only ``epoch``, the validation metrics and ``time``. Pass the list to
            ``pandas.DataFrame`` for a table.

        Raises:
            TypeError: If ``train_data`` is not a ``ReIDSplit``, or ``val_protocol`` is neither a
                ``RetrievalProtocol`` nor a ``ReassociationProtocol``.
            ValueError: If ``epochs`` or ``steps_per_epoch`` is below 1, ``warmup_epochs`` or
                ``freeze_backbone_epochs`` below 0, ``identities_per_batch`` or ``crops_per_identity`` below 2, or
                ``train_data`` holds fewer identities than ``identities_per_batch``. Also if ``train`` can't step
                ``lr_scheduler``, ``lr_scheduler_kwargs`` come without ``lr_scheduler``, ``trainable_when_frozen``
                names a module the backbone doesn't have, the augmentations change the crop's size, or
                ``output_dir`` holds another run's ``last/``, ``best/`` or ``metrics.csv``. With
                ``sample_per_sequence``, also if a crop's path doesn't name its sequence or no sequence holds 2
                identities. With ``val_data``, also if ``checkpoint_metric`` isn't a metric of ``val_protocol``, or,
                before the first epoch, if ``evaluate`` raises on the validation crops.
        """
        # Validate with RetrievalProtocol unless told otherwise
        if val_data is not None and val_protocol is None:
            val_protocol = RetrievalProtocol()

        # Check the arguments and the augmentations' crop size before any work
        if output_dir is not None:
            output_dir = Path(output_dir)
        self._check_train_arguments(
            train_data,
            output_dir,
            val_data,
            val_gallery,
            val_protocol,
            checkpoint_metric,
            epochs,
            steps_per_epoch,
            identities_per_batch,
            crops_per_identity,
            warmup_epochs,
            lr_scheduler,
            lr_scheduler_kwargs,
            freeze_backbone_epochs,
        )
        transform = build_train_transform(self._preprocessing, augmentations)

        # Pick the epoch by mAP for retrieval or MRR for re-association when checkpoint_metric is "auto"
        if val_data is not None and checkpoint_metric == "auto":
            checkpoint_metric = "map" if isinstance(val_protocol, RetrievalProtocol) else "mrr"

        # Seed torch for the augmentations and the new head, and random and numpy for transforms that use them
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Batches of identities_per_batch identities with crops_per_identity crops each
        identities, labels = np.unique(train_data.pids, return_inverse=True)
        groups = build_sequence_groups(train_data.image_paths, labels) if sample_per_sequence else None
        if steps_per_epoch is None:
            num_batches = max(1, len(train_data) // (identities_per_batch * crops_per_identity))
        else:
            num_batches = steps_per_epoch
        crops = CropDataset(train_data.image_paths, transform)
        sampler = PKSampler(labels, identities_per_batch, crops_per_identity, num_batches, seed, groups)
        loader = DataLoader(
            StackDataset(crops, torch.as_tensor(labels)),
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=self._device.type == "cuda",
            persistent_workers=num_workers > 0,
        )

        # The new classification head, the optimizer over it and the backbone, the schedule and mixed precision
        head = self._classification_head(len(identities))
        classification_loss.to(self._device)  # a loss may hold tensors, such as class weights
        embedding_loss.to(self._device)
        parameters = [*self._backbone.parameters(), *head.parameters()]
        optimizer_instance = optimizer(parameters, lr=lr, weight_decay=weight_decay, **(optimizer_kwargs or {}))
        schedule = build_lr_schedule_with_warmup(
            optimizer_instance, epochs, warmup_epochs, lr_scheduler, lr_scheduler_kwargs
        )
        use_mixed_precision = mixed_precision and self._device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision)

        # Record the run's settings before the first epoch, and the columns of metrics.csv
        if output_dir is not None:
            config = {
                "architecture": self._architecture_name,
                "val_protocol": None if val_data is None else repr(val_protocol),
                "checkpoint_metric": None if val_data is None else checkpoint_metric,
                "epochs": epochs,
                "steps_per_epoch": num_batches,
                "identities_per_batch": identities_per_batch,
                "crops_per_identity": crops_per_identity,
                "sample_per_sequence": sample_per_sequence,
                "lr": lr,
                "warmup_epochs": warmup_epochs,
                "weight_decay": weight_decay,
                "optimizer": optimizer.__name__,
                "optimizer_kwargs": optimizer_kwargs,
                "lr_scheduler": None if lr_scheduler is None else lr_scheduler.__name__,
                "lr_scheduler_kwargs": lr_scheduler_kwargs,
                "freeze_backbone_epochs": freeze_backbone_epochs,
                "classification_loss": repr(classification_loss),
                # torch's repr of CrossEntropyLoss leaves out label_smoothing
                "label_smoothing": getattr(classification_loss, "label_smoothing", None),
                "embedding_loss": repr(embedding_loss),
                "classification_loss_weight": classification_loss_weight,
                "embedding_loss_weight": embedding_loss_weight,
                "augmentations": [repr(augmentation) for augmentation in augmentations],
                "num_workers": num_workers,
                "mixed_precision": use_mixed_precision,
                "seed": seed,
                "num_identities": len(identities),
                "num_crops": len(train_data),
            }
            write_training_config(output_dir, config)
            columns = ["epoch", "lr", *LOSS_NAMES]
            if val_data is not None:
                columns += [f"val/{name}" for name in val_protocol.metric_names]
            columns.append("time")

        # Score the weights before training, recorded with epoch 0 and never picked as the best epoch
        history: list[dict[str, float]] = []
        if val_data is not None:
            start = time.perf_counter()
            scores = self._validate(val_data, val_gallery, val_protocol)
            history.append({"epoch": 0, **scores, "time": time.perf_counter() - start})
            if output_dir is not None:
                append_metrics_row(output_dir, history[0], columns)
            if verbose:
                print(format_epoch_line(history[0], epochs))

        # Train, saving last/ and a metrics.csv row after every epoch, and best/ after each epoch that becomes the best
        best_epoch = None
        best_score = -math.inf
        try:
            for epoch in range(1, epochs + 1):
                start = time.perf_counter()
                lr_this_epoch = optimizer_instance.param_groups[0]["lr"]
                losses = train_one_epoch(
                    self._backbone,
                    head,
                    loader,
                    optimizer_instance,
                    scaler,
                    device=self._device,
                    mixed_precision=use_mixed_precision,
                    classification_loss=classification_loss,
                    embedding_loss=embedding_loss,
                    classification_loss_weight=classification_loss_weight,
                    embedding_loss_weight=embedding_loss_weight,
                    freeze_backbone=epoch <= freeze_backbone_epochs,
                    description=f"Epoch {epoch}/{epochs}",
                    verbose=verbose,
                )
                schedule.step()

                # Save last/ before validating, so a crash during validation keeps the epoch's weights
                if output_dir is not None:
                    self.save(str(output_dir / "last"), verbose=False)

                # Score the epoch, and make it the best if no earlier epoch scored higher
                scores = {}
                if val_data is not None:
                    scores = self._validate(val_data, val_gallery, val_protocol)
                    if checkpoint_metric is not None and scores[f"val/{checkpoint_metric}"] >= best_score:
                        best_epoch = epoch
                        best_score = scores[f"val/{checkpoint_metric}"]
                        if output_dir is not None:
                            self.save(str(output_dir / "best"), verbose=False)

                row = {"epoch": epoch, "lr": lr_this_epoch, **losses, **scores, "time": time.perf_counter() - start}
                history.append(row)
                if output_dir is not None:
                    append_metrics_row(output_dir, row, columns)
                if verbose:
                    print(format_epoch_line(row, epochs))

            # Name the best epoch, and warn if the starting weights scored higher than every trained epoch
            if val_data is not None and checkpoint_metric is not None:
                where_saved = "" if output_dir is None else ", saved in best/"
                if verbose:
                    print(f"best epoch {best_epoch}  val/{checkpoint_metric} {best_score:.2f}{where_saved}")
                starting_score = history[0][f"val/{checkpoint_metric}"]
                if starting_score > best_score:
                    warnings.warn(
                        f"The weights before training scored val/{checkpoint_metric} {starting_score:.2f}, higher "
                        f"than every trained epoch, so training made the model worse on val_data. The best trained "
                        f"epoch is epoch {best_epoch}, at {best_score:.2f}{where_saved}.",
                        UserWarning,
                        stacklevel=2,
                    )
        finally:
            # cleaning state if there are errors like OOM
            self._backbone.requires_grad_(True)
            self._backbone.zero_grad()
            self._backbone.eval()
        return history

    def _validate(
        self, val_data: ReIDSplit, val_gallery: ReIDSplit | None, val_protocol: EvaluationProtocol
    ) -> dict[str, float]:
        """Score the model with ``evaluate`` in eval mode, and return its metrics named as ``val/`` columns.

        It doesn't affect the current random state.
        """
        self._backbone.eval()
        with torch.random.fork_rng(devices=[]):  # devices=[] forks only the CPU generator, the one a DataLoader uses
            result = self.evaluate(val_data, val_gallery, protocol=val_protocol, verbose=False)
        return {f"val/{name}": value for name, value in result.metrics.items()}

    def _classification_head(self, num_identities: int) -> nn.Linear:
        """Return a layer on the model's device that scores each of ``num_identities`` from an embedding.

        Its weights start small and random, and its bias at zero.
        """
        head = nn.Linear(self.embedding_dim, num_identities)
        nn.init.normal_(head.weight, std=0.01)
        nn.init.zeros_(head.bias)
        return head.to(self._device)

    def _check_train_arguments(
        self,
        train_data: ReIDSplit,
        output_dir: Path | None,
        val_data: ReIDSplit | None,
        val_gallery: ReIDSplit | None,
        val_protocol: EvaluationProtocol | None,
        checkpoint_metric: str | None,
        epochs: int,
        steps_per_epoch: int | None,
        identities_per_batch: int,
        crops_per_identity: int,
        warmup_epochs: int,
        lr_scheduler: type[LRScheduler] | None,
        lr_scheduler_kwargs: dict[str, object] | None,
        freeze_backbone_epochs: int,
    ) -> None:
        """Raise if the arguments of ``ReIDModel.train`` can't make a run, and warn about runs that would surprise."""
        if not isinstance(train_data, ReIDSplit):
            raise TypeError(
                f"train_data must be a ReIDSplit, got {type(train_data).__name__}. "
                "Load a folder of identity folders with load_identity_folders(root)."
            )
        if output_dir is not None:
            existing = [name for name in ("last", "best", "metrics.csv") if (output_dir / name).exists()]
            if existing:
                raise ValueError(
                    f"output_dir {output_dir} already holds {' and '.join(existing)} from another run. "
                    "Pass a new folder or remove them."
                )
        validate_positive_int(epochs, "epochs")
        if steps_per_epoch is not None:
            validate_positive_int(steps_per_epoch, "steps_per_epoch")
        # The embedding loss compares crops of one identity with crops of others, so a batch needs 2 of each.
        validate_int_at_least(identities_per_batch, "identities_per_batch", 2)
        validate_int_at_least(crops_per_identity, "crops_per_identity", 2)
        num_identities = len(np.unique(train_data.pids))
        if num_identities < identities_per_batch:
            raise ValueError(
                f"identities_per_batch={identities_per_batch} needs at least {identities_per_batch} identities in "
                f"train_data, got {num_identities}."
            )
        validate_int_at_least(warmup_epochs, "warmup_epochs", 0)
        if lr_scheduler is not None and issubclass(lr_scheduler, SCHEDULERS_TRAIN_CANNOT_STEP):
            raise ValueError(
                f"lr_scheduler={lr_scheduler.__name__} can't run in train, which steps the scheduler once per epoch "
                "with no score. ReduceLROnPlateau needs a validation score every epoch, OneCycleLR and CyclicLR step "
                "every batch and bring their own warmup, and ChainedScheduler and SequentialLR take schedulers already "
                "built on the optimizer, which train creates itself. Pass a scheduler stepped once per epoch, such as "
                "StepLR or CosineAnnealingLR."
            )
        if lr_scheduler is None and lr_scheduler_kwargs is not None:
            raise ValueError("lr_scheduler_kwargs needs an lr_scheduler to pass them to.")
        validate_int_at_least(freeze_backbone_epochs, "freeze_backbone_epochs", 0)
        child_names = {name for name, _ in self._backbone.named_children()}
        unknown = set(getattr(self._backbone, "trainable_when_frozen", ())) - child_names
        if unknown:
            raise ValueError(
                f"trainable_when_frozen names {sorted(unknown)}, which aren't direct child modules of the backbone."
            )
        if val_data is not None:
            if not isinstance(val_protocol, (RetrievalProtocol, ReassociationProtocol)):
                raise TypeError(
                    "val_protocol must be an instance, such as RetrievalProtocol() or ReassociationProtocol(), "
                    f"got {val_protocol!r}."
                )
            if checkpoint_metric not in (None, "auto", *val_protocol.metric_names):
                raise ValueError(
                    f"checkpoint_metric={checkpoint_metric!r} isn't a metric of {type(val_protocol).__name__}, which "
                    f"reports {', '.join(val_protocol.metric_names)}. Pass one of them, without the val/ prefix."
                )
        else:
            ignored = []
            if val_gallery is not None:
                ignored.append("val_gallery")
            if val_protocol is not None:
                ignored.append("val_protocol")
            if checkpoint_metric not in (None, "auto"):
                ignored.append("checkpoint_metric")
            if ignored:
                warnings.warn(
                    f"{' and '.join(ignored)} will be ignored without val_data. Pass val_data to validate.",
                    UserWarning,
                    stacklevel=3,
                )
        if warmup_epochs > epochs:
            warnings.warn(
                f"warmup_epochs={warmup_epochs} is more than epochs={epochs}, so the learning rate never reaches lr.",
                UserWarning,
                stacklevel=3,
            )
        if freeze_backbone_epochs >= epochs:
            warnings.warn(
                f"freeze_backbone_epochs={freeze_backbone_epochs} is at least epochs={epochs}, so the run trains only "
                "the classification layer and the backbone's trainable_when_frozen modules.",
                UserWarning,
                stacklevel=3,
            )

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
            gallery: Gallery images for the queries in ``data``. Retrieval ranks
                each query against all of them. Re-association ranks each query
                against the gallery crops of the same sequence ``gap_frames``
                later.
            protocol: Evaluation protocol. Defaults to
                ``RetrievalProtocol``.
            batch_size: Images per embedding batch.
            return_distmat: Keep the query-by-gallery distance matrix in the result.
            verbose: Log progress.

        Returns:
            ``ReIDResult`` with protocol metrics as percentages,
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
        row = {path: index for index, path in enumerate(paths)}
        query_embeddings = embeddings[[row[path] for path in query.image_paths]]
        gallery_embeddings = embeddings[[row[path] for path in gallery_split.image_paths]]

        distmat = distance_matrix(query_embeddings, gallery_embeddings, distance=protocol.distance)
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
        return embeddings.cpu().numpy().astype(np.float32)

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
            ValueError: If ``batch_size`` is not a positive integer.
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
            ValueError: If ``batch_size`` is not a positive integer.
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
