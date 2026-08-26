# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for ReID training losses."""

from __future__ import annotations

import pytest
import torch

from reid.training.losses import BatchHardTripletLoss, ReIDLoss


def _pk_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A PK-shaped batch: 8 crops over 4 identities, 8-dim embeddings, 4 logits."""
    embeddings = torch.randn(8, 8, requires_grad=True)
    logits = torch.randn(8, 4)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    return embeddings, logits, labels


class TestBatchHardTripletLoss:
    """Batch-hard triplet mining behavior."""

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
    """Combined CE + triplet (+ center) objective and its reported components."""

    def test_default_excludes_center_and_backprops(self) -> None:
        loss_fn = ReIDLoss(num_classes=4, feature_dim=8)
        embeddings, logits, labels = _pk_batch()

        total, components = loss_fn(embeddings, logits, labels)
        total.backward()

        assert set(components) == {"ce", "triplet", "total"}
        assert loss_fn.center is None
        # Embeddings reach the graph only through the triplet term.
        assert embeddings.grad.abs().sum() > 0

    def test_rejects_label_smoothing_outside_the_unit_interval(self) -> None:
        """torch accepts a negative label_smoothing silently; ReIDLoss must not."""
        for bad in (-0.1, 1.0):
            with pytest.raises(ValueError, match="label_smoothing must be in"):
                ReIDLoss(num_classes=4, feature_dim=8, label_smoothing=bad)

    def test_rejects_logits_that_do_not_match_num_classes(self) -> None:
        """Smoothing spreads over the logits width, so a mismatch must not pass silently."""
        loss_fn = ReIDLoss(num_classes=10, feature_dim=8)
        embeddings, _, labels = _pk_batch()

        with pytest.raises(ValueError, match="logits must be"):
            loss_fn(embeddings, torch.randn(8, 6), labels)

    def test_center_adds_a_weighted_term_and_trainable_centers(self) -> None:
        loss_fn = ReIDLoss(num_classes=4, feature_dim=8, use_center=True, center_weight=0.1)
        embeddings, logits, labels = _pk_batch()

        total, components = loss_fn(embeddings, logits, labels)

        assert set(components) == {"ce", "triplet", "center", "total"}
        expected = components["ce"] + loss_fn.triplet_weight * components["triplet"] + 0.1 * components["center"]
        assert total.item() == pytest.approx(expected, rel=1e-5)

        total.backward()
        assert loss_fn.center.centers.grad is not None
