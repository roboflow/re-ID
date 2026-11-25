# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""``TripletLoss``: batch-hard mining on embeddings scaled to length 1."""

from __future__ import annotations

import pytest
import torch

from reid.training.losses import TripletLoss


def test_loss_uses_the_farthest_positive() -> None:
    axes = torch.eye(4)
    embeddings = torch.cat([axes, -axes])  # +x, +y, +z, +w, then -x, -y, -z, -w
    labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])  # person 0 on a square in the x-y plane, person 1 in z-w

    loss = TripletLoss(margin=0.3)(embeddings, labels)

    # For every crop, the opposite corner of its square is 2 away and the neighbours sqrt(2), and the other
    # person's square is at right angles, so all of its crops are sqrt(2) away
    farthest_positive = 2.0
    nearest_negative = 2**0.5
    assert loss.item() == pytest.approx(0.3 + farthest_positive - nearest_negative)


def test_loss_uses_the_nearest_negative() -> None:
    # Person 0 sits at 0 and 180 degrees on the unit circle, and person 1 at 60 and 240 degrees
    embeddings = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.5, 3**0.5 / 2], [-0.5, -(3**0.5) / 2]])
    labels = torch.tensor([0, 0, 1, 1])

    loss = TripletLoss(margin=0.3)(embeddings, labels)

    # For every crop, its positive is across the circle, 2 away, and the other person's crops are 60 degrees
    # away at 1 (an equilateral triangle with the centre) and 120 degrees away at sqrt(3)
    farthest_positive = 2.0
    nearest_negative = 1.0
    assert loss.item() == pytest.approx(0.3 + farthest_positive - nearest_negative)


def test_loss_ignores_the_length_of_the_embeddings() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])  # length 1
    labels = torch.tensor([0, 0, 1, 1])
    longer = embeddings * torch.tensor([[2.0], [3.0], [4.0], [5.0]])  # same directions, lengths 2 to 5

    triplet_loss = TripletLoss(margin=0.3)

    assert triplet_loss(longer, labels).item() == pytest.approx(triplet_loss(embeddings, labels).item())


def test_loss_is_zero_when_every_negative_is_beyond_the_margin() -> None:
    # Positives are 0.63 apart and the nearest negative is 1.9 away, more than 0.63 + 0.3.
    embeddings = torch.tensor([[1.0, 0.0], [0.8, 0.6], [-1.0, 0.0], [-0.8, -0.6]])
    labels = torch.tensor([0, 0, 1, 1])

    loss = TripletLoss(margin=0.3)(embeddings, labels)

    assert loss.item() == 0.0
