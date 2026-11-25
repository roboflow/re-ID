# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""PK batch sampling for re-ID training."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler

from reid.data.identity_crops import _indices_by_label


class PKSampler(Sampler[list[int]]):
    """Yields batches of ``identities_per_batch`` identities with ``crops_per_identity`` crops each.

    Each batch draws ``identities_per_batch`` different identities uniformly at random. Then it takes
    ``crops_per_identity`` crops of each without repetition within the epoch, from a shuffled order of that
    identity's crops that every epoch starts anew. When fewer than ``crops_per_identity`` remain in the order, they
    are skipped and a new order is shuffled, so an identity drawn more often than its crops allow repeats some. An
    identity with fewer crops than ``crops_per_identity`` repeats some in each batch. One generator, seeded once,
    draws every batch, so each epoch gets new ones.

    Batches of P identities with K crops each, from:
        Hermans et al. *In Defense of the Triplet Loss for Person Re-Identification*. arXiv:1703.07737, 2017.

    Args:
        labels: Class of each crop.
        identities_per_batch: Identities in a batch (P).
        crops_per_identity: Crops of each identity in a batch (K).
        num_batches: Batches per epoch.
        seed: Seed for the generator.
    """

    def __init__(
        self,
        labels: Sequence[int],
        identities_per_batch: int,
        crops_per_identity: int,
        num_batches: int,
        seed: int,
    ) -> None:
        self._indices_by_label = _indices_by_label(labels)
        self._identities_per_batch = identities_per_batch
        self._crops_per_identity = crops_per_identity
        self._num_batches = num_batches
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[list[int]]:
        identities = list(self._indices_by_label.keys())
        # Each identity's crops not used yet this epoch, in a shuffled order
        unused_crops: dict[int, list[int]] = {}
        for _ in range(self._num_batches):
            batch: list[int] = []
            # Sample random identities for the batch
            for identity in self._rng.sample(identities, self._identities_per_batch):
                batch += self._next_crops(identity, unused_crops)
            yield batch

    def _next_crops(self, identity: int, unused_crops: dict[int, list[int]]) -> list[int]:
        """The next ``crops_per_identity`` crops of ``identity`` in its shuffled order, reshuffled once fewer remain."""
        crops = self._indices_by_label[identity]
        if len(crops) < self._crops_per_identity:  # fewer crops than K, so some repeat
            return self._rng.choices(crops, k=self._crops_per_identity)
        unused = unused_crops.get(identity, [])
        if len(unused) < self._crops_per_identity:  # start a new shuffled order of all its crops
            unused = self._rng.sample(crops, len(crops))
        unused_crops[identity] = unused[self._crops_per_identity :]
        return unused[: self._crops_per_identity]
