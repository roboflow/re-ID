# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""ReID fine-tuning losses (label-smoothed CE, batch-hard triplet, optional center)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class BatchHardTripletLoss(nn.Module):
    """Batch-hard triplet loss (hardest positive/negative per anchor).

    For each anchor, the hardest positive is the farthest same-identity sample
    and the hardest negative is the nearest different-identity sample.
    """

    def __init__(self, margin: float = 0.3, normalize: bool = True) -> None:
        super().__init__()
        self.margin = margin
        self.normalize = normalize

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute batch-hard triplet loss."""
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        distances = torch.cdist(embeddings, embeddings, p=2).clamp(min=1e-12)
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        different = ~same
        positive_mask = same.clone()
        positive_mask.fill_diagonal_(False)

        hardest_positive = (distances * positive_mask).max(dim=1).values
        masked = distances.clone()
        masked[~different] = float("inf")
        hardest_negative = masked.min(dim=1).values

        # ranking the mined pairs is max(0, margin + positive - negative)
        return F.margin_ranking_loss(
            hardest_negative,
            hardest_positive,
            torch.ones_like(hardest_positive),
            margin=self.margin,
        )


class CenterLoss(nn.Module):
    """Center loss: pull embeddings toward learnable class centers.

    Centers start from ``randn`` and are learned by the caller's optimizer, so
    they only become meaningful class prototypes once they have had enough steps
    to travel from that initialisation onto the embedding manifold.
    """

    def __init__(self, num_classes: int, feature_dim: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute center loss."""
        centers_batch = self.centers.index_select(0, labels)
        # Wen et al.: mean over the batch of half the squared distance to the class center.
        return F.mse_loss(embeddings, centers_batch, reduction="sum") / (2 * embeddings.size(0))


class ReIDLoss(nn.Module):
    """Combined CE + triplet (+ optional center) ReID objective."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        *,
        margin: float = 0.3,
        label_smoothing: float = 0.1,
        triplet_weight: float = 1.0,
        center_weight: float = 0.0005,
        use_center: bool = False,
        normalize_triplet: bool = True,
    ) -> None:
        super().__init__()
        # torch accepts negative label_smoothing silently; keep the tighter range.
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}.")
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.triplet = BatchHardTripletLoss(margin=margin, normalize=normalize_triplet)
        self.triplet_weight = triplet_weight
        self.center_weight = center_weight if use_center else 0.0
        self.center = CenterLoss(num_classes, feature_dim) if use_center else None

    def forward(
        self,
        embeddings: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return ``(total_loss, component_dict)``."""
        ce = self.cross_entropy(logits, labels)
        triplet = self.triplet(embeddings, labels)
        total = ce + self.triplet_weight * triplet
        components = {"ce": ce.item(), "triplet": triplet.item()}

        if self.center is not None:
            center = self.center(embeddings, labels)
            total = total + self.center_weight * center
            components["center"] = center.item()

        components["total"] = total.item()
        return total, components
