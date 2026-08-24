# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Fine-tune a ReID encoder on a per-identity crop dataset."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
from torch import nn
from torch.utils.data import DataLoader

from reid.architectures import build_architecture
from reid.data import ReIDSplit, load_identity_crops
from reid.loaders import load_state_dict, resolve_load_plan, resolve_weights
from reid.model import ReIDModel, _select_device
from reid.preprocessing import ReIDPreprocessing
from reid.training.datasets import (
    IdentitiesPerEpoch,
    PKSampler,
    ReIDTrainDataset,
    _resolve_pk_num_batches,
    build_eval_transform,
    build_train_transform,
)
from reid.training.losses import ReIDLoss

logger = logging.getLogger(__name__)


@runtime_checkable
class TrainableBackbone(Protocol):
    """Contract a backbone must satisfy to be fine-tuned by :func:`train_reid`.

    Satisfied by :class:`~reid.architectures.osnet.OSNet` and
    :class:`~reid.architectures.fastreid_sbs.FastReIDSBSResNeSt50` when built
    with ``num_classes > 0``.
    """

    feature_dim: int

    def forward_embeddings_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(embeddings, logits)`` for a batch of crops."""
        ...

    def set_backbone_frozen(self, frozen: bool) -> None:
        """Freeze or unfreeze the trunk, keeping the training head trainable."""
        ...


@dataclass
class TrainConfig:
    """Hyper-parameters for :func:`train_reid`.

    Args:
        epochs: Training epochs.
        p: Identities per PK batch.
        k: Instances per identity per PK batch.
        lr: Adam learning rate.
        weight_decay: Adam weight decay.
        warmup_epochs: Linear LR warmup length.
        lr_milestones: Epochs where LR is multiplied by ``lr_gamma``.
        lr_gamma: Multiplicative LR decay at each milestone.
        margin: Batch-hard triplet margin.
        label_smoothing: Cross-entropy label smoothing.
        triplet_weight: Weight on the triplet term.
        use_center: Enable the optional center loss term.
        center_weight: Weight on the center term in the total loss. Controls how
            hard the term pulls embeddings toward their class center; it does not
            affect how the centers themselves are learned (see ``center_lr``).
        center_lr: Adam learning rate for the center parameters, which ride in a
            dedicated group of the main optimizer. Bag of Tricks instead trains
            centers with SGD and rescales their gradients by ``1 / center_weight``;
            Adam normalizes gradient magnitude, so ``center_weight`` cancels out of
            the center update and only ``center_lr`` governs how fast centers move.
        freeze_backbone_epochs: Freeze backbone (except head) for this many epochs.
        num_workers: DataLoader workers.
        amp: Use CUDA autocast + GradScaler when on GPU.
        device: Torch device string, or ``"auto"``.
        seed: RNG seed.
        log_interval: Log every N optimizer steps.
        identities_per_epoch: Identity slots to cover per epoch. ``"all"`` uses
            every training identity. An int may be smaller (smoke runs) or larger
            (multiple passes) than the dataset size.
        crops_per_identity: Crop draws per identity slot per epoch. ``None`` uses
            ``k``. Epoch batches are
            ``(identities_per_epoch * crops_per_identity) // (p * k)``.
    """

    epochs: int = 30
    p: int = 8
    k: int = 4
    lr: float = 3e-4
    weight_decay: float = 5e-4
    warmup_epochs: int = 5
    lr_milestones: tuple[int, ...] = (20, 27)
    lr_gamma: float = 0.1
    margin: float = 0.3
    label_smoothing: float = 0.1
    triplet_weight: float = 1.0
    use_center: bool = False
    center_weight: float = 0.0005
    center_lr: float = 0.5
    freeze_backbone_epochs: int = 0
    num_workers: int = 4
    amp: bool = True
    device: str = "auto"
    seed: int = 42
    log_interval: int = 20
    identities_per_epoch: IdentitiesPerEpoch = "all"
    crops_per_identity: int | None = None

    def __post_init__(self) -> None:
        if self.identities_per_epoch != "all" and (
            not isinstance(self.identities_per_epoch, int) or self.identities_per_epoch < 1
        ):
            raise ValueError(f"identities_per_epoch must be 'all' or an int >= 1, got {self.identities_per_epoch!r}.")
        if self.crops_per_identity is not None and self.crops_per_identity < 1:
            raise ValueError(f"crops_per_identity must be >= 1, got {self.crops_per_identity}.")
        if self.log_interval < 0:
            raise ValueError(f"log_interval must be >= 0, got {self.log_interval}.")
        if self.triplet_weight > 0 and (self.p < 2 or self.k < 2):
            raise ValueError(
                "Batch-hard triplet mining needs p >= 2 and k >= 2 "
                f"(got p={self.p}, k={self.k}, triplet_weight={self.triplet_weight})."
            )

    def num_batches(self, num_identities: int) -> int:
        """PK batches per epoch for a dataset with ``num_identities`` labels."""
        return _resolve_pk_num_batches(
            num_identities,
            self.p,
            self.k,
            identities_per_epoch=self.identities_per_epoch,
            crops_per_identity=self.k if self.crops_per_identity is None else self.crops_per_identity,
        )


@dataclass
class TrainResult:
    """Output of :func:`train_reid` (model, class count, history, save path)."""

    model: ReIDModel
    num_classes: int
    history: list[dict[str, float]] = field(default_factory=list)
    output_dir: str | None = None
    best_epoch: int | None = None
    best_val_total: float | None = None


def _build_lr_lambda(config: TrainConfig) -> Callable[[int], float]:
    def lr_lambda(epoch: int) -> float:
        if config.warmup_epochs > 0 and epoch < config.warmup_epochs:
            return float(epoch + 1) / float(config.warmup_epochs)
        factor = 1.0
        for milestone in config.lr_milestones:
            if epoch >= milestone:
                factor *= config.lr_gamma
        return factor

    return lr_lambda


def _resolve_training_recipe(
    architecture: str | None,
    pretrained: str | None,
    preprocessing: ReIDPreprocessing | None,
) -> tuple[str | nn.Module, str | None, ReIDPreprocessing]:
    """Resolve architecture, warm-start weights, and preprocessing.

    Delegates to :func:`reid.loaders.resolve_load_plan` so training shares the
    single load-recipe entry point with ``ReIDModel.from_pretrained``. The
    plan's strict-load fraction is not applied here: the trainable backbone
    carries a classifier head that inference checkpoints do not contain, so the
    warm-start is a deliberately partial load (see :func:`_build_trainable_backbone`).
    """
    plan = resolve_load_plan(pretrained, architecture, preprocessing)
    return plan.architecture, plan.weights, plan.preprocessing


def _build_trainable_backbone(
    arch_name: str | nn.Module,
    weights_source: str | None,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    """Build a backbone with a classifier head and optional warm-start weights.

    Raises:
        TypeError: If the architecture does not satisfy :class:`TrainableBackbone`.
    """
    backbone = build_architecture(arch_name, num_classes=num_classes)
    if not isinstance(backbone, TrainableBackbone):
        raise TypeError(
            "train_reid needs a backbone implementing forward_embeddings_logits(images), "
            "set_backbone_frozen(frozen), and feature_dim. Supported: OSNet, FastReID SBS. "
            f"Got {type(backbone).__name__}."
        )
    if weights_source is not None:
        local_path = resolve_weights(weights_source)
        # Warm-start is a partial load by design: the classifier head is fresh,
        # so a couple of keys always miss. The 0.5 floor still rejects a
        # checkpoint that does not match the architecture at all.
        load_state_dict(
            backbone,
            local_path,
            device,
            architecture=arch_name if isinstance(arch_name, str) else None,
            required_match_fraction=0.5,
        )
        logger.info("Warm-start from %s", weights_source)
    return backbone


def _clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _run_epoch(
    *,
    backbone: nn.Module,
    loss_fn: ReIDLoss,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    log_interval: int,
    epoch: int,
    step: int,
    frozen_backbone: bool = False,
) -> tuple[dict[str, float], int]:
    """Run one train or eval epoch. When ``optimizer`` is set, apply gradients."""
    train_mode = optimizer is not None
    if train_mode:
        backbone.train()
        # ``backbone.train()`` would undo trunk ``eval()`` from freeze; re-apply.
        if frozen_backbone:
            backbone.set_backbone_frozen(True)
    else:
        backbone.eval()

    epoch_totals: dict[str, float] = {}
    num_batches = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            embeddings, logits = backbone.forward_embeddings_logits(images)
            loss, components = loss_fn(embeddings, logits, labels)

        if optimizer is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if log_interval > 0 and step % log_interval == 0:
                logger.info(
                    "epoch %d step %d: %s",
                    epoch,
                    step,
                    {k: round(v, 4) for k, v in components.items()},
                )

        for key, value in components.items():
            epoch_totals[key] = epoch_totals.get(key, 0.0) + value
        num_batches += 1

    epoch_mean = {k: v / max(1, num_batches) for k, v in epoch_totals.items()}
    return epoch_mean, step


def _build_val_loader(
    val_data: ReIDSplit | str | Path,
    train_dataset: ReIDTrainDataset,
    config: TrainConfig,
    preprocessing: ReIDPreprocessing,
    device: torch.device,
) -> DataLoader:
    """Build a PK-sampled val loader over crops sharing train identity folders."""
    val_split = val_data if isinstance(val_data, ReIDSplit) else load_identity_crops(val_data)
    eval_transform = build_eval_transform(preprocessing)
    val_dataset = ReIDTrainDataset.from_identity_label_map(
        val_split,
        train_dataset.label_by_identity,
        transform=eval_transform,
        num_classes=train_dataset.num_classes,
    )
    n_val_ids = len(set(val_dataset.labels))
    if n_val_ids < config.p:
        raise ValueError(f"Validation needs at least p={config.p} overlapping identities with train, got {n_val_ids}.")
    logger.info(
        "Val crops (overlapping identities): %d identities, %d crops",
        n_val_ids,
        len(val_dataset),
    )
    val_sampler = PKSampler(
        val_dataset.labels,
        p=config.p,
        k=config.k,
        seed=config.seed,
        identities_per_epoch=config.identities_per_epoch,
        crops_per_identity=config.crops_per_identity,
    )
    return DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )


def train_reid(
    data: ReIDSplit | str | Path,
    config: TrainConfig | None = None,
    *,
    architecture: str | None = None,
    pretrained: str | None = None,
    preprocessing: ReIDPreprocessing | None = None,
    output_dir: str | Path | None = None,
    val_data: ReIDSplit | str | Path | None = None,
) -> TrainResult:
    """Fine-tune a ReID encoder on a crop :class:`ReIDSplit`.

    Args:
        data: A crop :class:`ReIDSplit`, or a path to an identity-folder crop
            dataset. A path is loaded via :func:`reid.data.load_identity_crops`;
            filter identities on the loader when you need a subset.
        config: Training hyper-parameters.
        architecture: Backbone name override. When set with ``pretrained=None``,
            the backbone is randomly initialised (no warm-start).
        pretrained: Weights source / alias to warm-start from. When
            ``architecture`` is omitted, ``None`` resolves to the default curated
            alias (warm-start included). When ``architecture`` is set, ``None``
            means no warm-start weights.
        preprocessing: Preprocessing override.
        output_dir: If set, save via :meth:`ReIDModel.save_pretrained`.
        val_data: Optional val crop split/path. Crops whose identity folder names
            overlap the training set contribute CE + triplet validation loss each
            epoch. When set, the checkpoint written to ``output_dir`` is the epoch
            with the lowest ``val_total``.

    Returns:
        :class:`TrainResult`.
    """
    config = config or TrainConfig()
    device = _select_device(config.device)
    torch.manual_seed(config.seed)

    split = data if isinstance(data, ReIDSplit) else load_identity_crops(data)

    arch_name, weights_source, resolved_preprocessing = _resolve_training_recipe(
        architecture, pretrained, preprocessing
    )
    transform = build_train_transform(resolved_preprocessing)
    dataset = ReIDTrainDataset(split, transform=transform)
    num_classes = dataset.num_classes
    logger.info("Crop dataset: %d identities, %d crops", num_classes, len(dataset))

    sampler = PKSampler(
        dataset.labels,
        p=config.p,
        k=config.k,
        seed=config.seed,
        identities_per_epoch=config.identities_per_epoch,
        crops_per_identity=config.crops_per_identity,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = None
    if val_data is not None:
        val_loader = _build_val_loader(val_data, dataset, config, resolved_preprocessing, device)

    backbone = _build_trainable_backbone(arch_name, weights_source, num_classes, device)
    backbone.to(device).train()
    loss_fn = ReIDLoss(
        num_classes,
        int(backbone.feature_dim),
        margin=config.margin,
        label_smoothing=config.label_smoothing,
        triplet_weight=config.triplet_weight,
        use_center=config.use_center,
        center_weight=config.center_weight,
    ).to(device)

    param_groups: list[dict] = [{"params": backbone.parameters()}]
    if loss_fn.center is not None:
        param_groups.append(
            {
                "params": loss_fn.center.parameters(),
                "lr": config.center_lr,
                "weight_decay": 0.0,
            }
        )
    optimizer = torch.optim.Adam(param_groups, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _build_lr_lambda(config))

    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict[str, float]] = []
    step = 0
    best_epoch: int | None = None
    best_val_total: float | None = None
    best_backbone_state: dict[str, torch.Tensor] | None = None
    best_center_state: dict[str, torch.Tensor] | None = None

    for epoch in range(config.epochs):
        sampler.set_epoch(epoch)
        frozen = epoch < config.freeze_backbone_epochs
        backbone.set_backbone_frozen(frozen)

        epoch_mean, step = _run_epoch(
            backbone=backbone,
            loss_fn=loss_fn,
            loader=loader,
            device=device,
            use_amp=use_amp,
            optimizer=optimizer,
            scaler=scaler,
            log_interval=config.log_interval,
            epoch=epoch,
            step=step,
            frozen_backbone=frozen,
        )
        epoch_mean["epoch"] = epoch
        # Record before scheduler.step() so this is the LR the epoch trained at.
        epoch_mean["lr"] = optimizer.param_groups[0]["lr"]
        scheduler.step()

        if val_loader is not None:
            val_sampler = val_loader.batch_sampler
            if hasattr(val_sampler, "set_epoch"):
                val_sampler.set_epoch(epoch)
            with torch.no_grad():
                val_mean, _ = _run_epoch(
                    backbone=backbone,
                    loss_fn=loss_fn,
                    loader=val_loader,
                    device=device,
                    use_amp=use_amp,
                    optimizer=None,
                    scaler=scaler,
                    log_interval=config.log_interval,
                    epoch=epoch,
                    step=step,
                )
            epoch_mean["val_ce"] = val_mean["ce"]
            epoch_mean["val_triplet"] = val_mean["triplet"]
            epoch_mean["val_total"] = val_mean["total"]
            if "center" in val_mean:
                epoch_mean["val_center"] = val_mean["center"]

            if best_val_total is None or val_mean["total"] < best_val_total:
                best_val_total = val_mean["total"]
                best_epoch = epoch
                best_backbone_state = _clone_state_dict(backbone)
                if loss_fn.center is not None:
                    best_center_state = _clone_state_dict(loss_fn.center)

        history.append(epoch_mean)
        logger.info("epoch %d mean: %s", epoch, {k: round(v, 4) for k, v in epoch_mean.items()})

    if best_backbone_state is not None:
        backbone.load_state_dict(best_backbone_state)
        if best_center_state is not None and loss_fn.center is not None:
            loss_fn.center.load_state_dict(best_center_state)
        logger.info(
            "Restored best validation epoch %s (val_total=%.4f)",
            best_epoch,
            best_val_total,
        )

    backbone.eval()
    model = ReIDModel(
        backbone,
        device,
        resolved_preprocessing,
        architecture=arch_name if isinstance(arch_name, str) else None,
    )

    saved_dir = None
    if output_dir is not None:
        saved_dir = str(output_dir)
        model.save_pretrained(saved_dir)

    return TrainResult(
        model=model,
        num_classes=num_classes,
        history=history,
        output_dir=saved_dir,
        best_epoch=best_epoch,
        best_val_total=best_val_total,
    )
