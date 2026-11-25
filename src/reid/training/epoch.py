# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""One training epoch of a re-ID encoder and its classification head."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

LOSS_NAMES = ("train/loss", "train/classification_loss", "train/embedding_loss")


def train_one_epoch(
    backbone: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    mixed_precision: bool,
    classification_loss: nn.Module,
    embedding_loss: nn.Module,
    classification_loss_weight: float,
    embedding_loss_weight: float,
    freeze_backbone: bool,
    description: str,
    verbose: bool,
) -> dict[str, float]:
    """Train ``backbone`` and ``head`` on every batch of ``loader`` once.

    Args:
        backbone: Encoder returning ``(B, D)`` embeddings.
        head: Maps embeddings to one score per training identity.
        loader: Batches of ``(images, labels)``.
        optimizer: Optimizer over the backbone and head parameters.
        scaler: Gradient scaler, enabled when ``mixed_precision`` is. It scales the loss up so float16 gradients
            don't round to zero, and the gradients back down before the optimizer step.
        device: Device of ``backbone`` and ``head``, where each batch is moved.
        mixed_precision: Run the forward pass under CUDA autocast.
        classification_loss: Loss on the head's scores.
        embedding_loss: Loss on the embeddings.
        classification_loss_weight: Weight of the classification loss in the total.
        embedding_loss_weight: Weight of the embedding loss in the total.
        freeze_backbone: Train only the head and the backbone's ``trainable_when_frozen`` modules.
        description: Progress bar label.
        verbose: Show the progress bar.

    Returns:
        The epoch's mean ``train/loss`` (the weighted total), ``train/classification_loss`` and
        ``train/embedding_loss`` (both unweighted).
    """
    _set_backbone_frozen(backbone, freeze_backbone)
    totals = {name: 0.0 for name in LOSS_NAMES}

    progress = tqdm(loader, desc=description, disable=not verbose)
    for step, (images, labels) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", enabled=mixed_precision):
            embeddings = backbone(images)
            classification = classification_loss(head(embeddings), labels)
            embedding = embedding_loss(embeddings, labels)
            loss = classification_loss_weight * classification + embedding_loss_weight * embedding
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        totals["train/loss"] += loss.item()
        totals["train/classification_loss"] += classification.item()
        totals["train/embedding_loss"] += embedding.item()
        progress.set_postfix(loss=f"{totals['train/loss'] / step:.2f}")

    return {name: total / len(loader) for name, total in totals.items()}


def _set_backbone_frozen(backbone: nn.Module, frozen: bool) -> None:
    """Put ``backbone`` in train mode and set which of its modules train.

    When ``frozen``, only the direct child modules named in ``trainable_when_frozen`` train. The others get no
    gradients and go to eval mode. Otherwise, every module trains.
    """
    trainable_when_frozen = getattr(backbone, "trainable_when_frozen", ())
    backbone.train()
    for name, parameter in backbone.named_parameters():
        module_name = name.split(".", 1)[0]
        parameter.requires_grad = not frozen or module_name in trainable_when_frozen
    if frozen:
        for name, module in backbone.named_children():
            if name not in trainable_when_frozen:
                module.eval()
