# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Embedding losses for re-ID training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TripletLoss(nn.Module):
    """Batch-hard triplet loss on embeddings scaled to length 1.

    For each crop in the batch, the hardest positive is the farthest crop of the same person and the hardest negative
    the nearest crop of another person, by Euclidean distance. The loss is the batch mean of
    ``max(0, margin + positive_distance - negative_distance)``.

    Loss from:
        Hermans et al. *In Defense of the Triplet Loss for Person Re-Identification*. arXiv:1703.07737, 2017.

    Args:
        margin: How much nearer the hardest positive must be than the hardest negative.
    """

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def extra_repr(self) -> str:
        return f"margin={self.margin}"

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the loss for a batch.

        Args:
            embeddings: Embeddings of shape ``(B, D)``.
            labels: Class of each embedding, shape ``(B,)``.

        Returns:
            Scalar loss.
        """
        embeddings = F.normalize(embeddings, dim=1)
        distances = torch.cdist(embeddings, embeddings).clamp(min=1e-12)
        same_person = labels.unsqueeze(0) == labels.unsqueeze(1)
        hardest_positive = torch.where(same_person, distances, 0.0).max(dim=1).values
        hardest_negative = torch.where(same_person, float("inf"), distances).min(dim=1).values
        return F.relu(self.margin + hardest_positive - hardest_negative).mean()
