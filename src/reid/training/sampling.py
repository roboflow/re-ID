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

    Each batch draws ``identities_per_batch`` different identities uniformly at random, so an identity with many
    crops is drawn as often as one with few. Then it draws ``crops_per_identity`` crops of each, repeating some when
    the identity has fewer. One generator, seeded once, draws every batch, so each epoch gets new ones.

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
        for _ in range(self._num_batches):
            batch: list[int] = []
            # Sample random identities for the batch
            for identity in self._rng.sample(identities, self._identities_per_batch):
                crops = self._indices_by_label[identity]
                if len(crops) >= self._crops_per_identity:
                    batch += self._rng.sample(crops, self._crops_per_identity)  # Sample which crops for the identity
                else:  # fewer crops than K, so some repeat
                    batch += self._rng.choices(crops, k=self._crops_per_identity)
            yield batch
