# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for ReID training losses."""

from __future__ import annotations

import torch

from reid.training.losses import BatchHardTripletLoss, ReIDLoss


class TestBatchHardTripletLoss:
    """Tests for :class:`BatchHardTripletLoss`."""

    def test_zero_for_well_separated_clusters(self) -> None:
        loss_fn = BatchHardTripletLoss(margin=0.3, normalize=False)
        embeddings = torch.tensor([[0.0, 0.0], [1.0, 0.0], [50.0, 0.0], [51.0, 0.0]])
        labels = torch.tensor([0, 0, 1, 1])
        assert loss_fn(embeddings, labels).item() == 0.0

    def test_positive_when_negatives_are_closer(self) -> None:
        loss_fn = BatchHardTripletLoss(margin=0.3, normalize=False)
        embeddings = torch.tensor([[0.0, 0.0], [10.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        labels = torch.tensor([0, 0, 1, 1])
        assert loss_fn(embeddings, labels).item() > 0.0


class TestReIDLoss:
    """Tests for :class:`ReIDLoss`."""

    def test_default_excludes_center_and_backprops(self) -> None:
        loss_fn = ReIDLoss(num_classes=4, feature_dim=8)
        embeddings = torch.randn(8, 8, requires_grad=True)
        logits = torch.randn(8, 4)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        total, components = loss_fn(embeddings, logits, labels)

        assert set(components) == {"ce", "triplet", "total"}
        assert loss_fn.center is None
        total.backward()
        assert embeddings.grad is not None
