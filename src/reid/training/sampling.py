# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""PK batch sampling for re-ID training, and the per-sequence groups it can draw from."""

from __future__ import annotations

import random
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
from torch.utils.data import Sampler

from reid.data.identity_crops import _indices_by_label
from reid.data.mot import _parse_crop_path


class PKSampler(Sampler[list[int]]):
    """Yields batches of ``identities_per_batch`` identities with ``crops_per_identity`` crops each.

    Each batch draws ``identities_per_batch`` different identities uniformly at random. Then it takes
    ``crops_per_identity`` crops of each without repetition within the epoch, from a shuffled order of that
    identity's crops that every epoch starts anew. When fewer than ``crops_per_identity`` remain in the order, they
    are skipped and a new order is shuffled, so an identity drawn more often than its crops allow repeats some. An
    identity with fewer crops than ``crops_per_identity`` repeats some in each batch. One generator, seeded once,
    draws every batch, so each epoch gets new ones.

    With ``groups``, every batch comes from one group, such as the identities of one tracking sequence, so the
    negatives each crop is compared with in the batch are people of the same group. A batch picks a group, then
    ``identities_per_batch`` of its identities at random. A large group is picked more often, in proportion to its
    number of identities, but each pick takes only ``identities_per_batch`` of them, so a person in a small group is
    drawn as often as one in a large group. A group with fewer identities than ``identities_per_batch`` gives a
    smaller batch with all of them, and is picked as if it had ``identities_per_batch``, since each pick already draws
    all of its identities.

    Batches of P identities with K crops each, from:
        Hermans et al. *In Defense of the Triplet Loss for Person Re-Identification*. arXiv:1703.07737, 2017.

    Args:
        labels: Class of each crop.
        identities_per_batch: Identities in a batch (P).
        crops_per_identity: Crops of each identity in a batch (K).
        num_batches: Batches per epoch.
        seed: Seed for the generator.
        groups: Identities that may share a batch, each group with at least 2. ``None`` puts every identity in one
            group, so batches mix the whole set.
    """

    def __init__(
        self,
        labels: Sequence[int],
        identities_per_batch: int,
        crops_per_identity: int,
        num_batches: int,
        seed: int,
        groups: Sequence[Sequence[int]] | None = None,
    ) -> None:
        self._indices_by_label = _indices_by_label(labels)
        self._identities_per_batch = identities_per_batch
        self._crops_per_identity = crops_per_identity
        self._num_batches = num_batches
        self._rng = random.Random(seed)

        # Without groups, every identity is in one group, so a batch draws from the whole set
        every_identity = list(self._indices_by_label.keys())
        self._groups = [every_identity] if groups is None else groups
        # Weight each group so that every identity has the same chance to be drawn in a batch
        self._weights = [max(len(identities), identities_per_batch) for identities in self._groups]

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[list[int]]:
        # Each identity's crops not used yet this epoch, in a shuffled order
        unused_crops: dict[int, list[int]] = {}
        for _ in range(self._num_batches):
            batch: list[int] = []
            # Sample a random group, then random identities of it for the batch
            identities = self._rng.choices(self._groups, self._weights)[0]
            for identity in self._rng.sample(identities, min(self._identities_per_batch, len(identities))):
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


def build_sequence_groups(image_paths: Sequence[str], labels: np.ndarray) -> list[list[int]]:
    """Group the identities by sequence, so that ``PKSampler`` can draw each batch from one sequence.

    Each identity's sequence is read from its first crop's path, ``{sequence}_{track}/{sequence}_{frame}.jpg`` as
    ``generate_mot_patches`` writes it. Sequences with a single identity are left out, with a warning, since a batch
    needs 2 identities.

    Args:
        image_paths: Path of each crop.
        labels: Identity of each crop, numbered from 0.

    Returns:
        The identities of each sequence that holds at least 2.

    Raises:
        ValueError: If a path doesn't name its sequence, or no sequence holds 2 identities.
    """
    # Read each identity's sequence from the path of its first crop
    sequence_of_identity: dict[int, str] = {}
    for path, label in zip(image_paths, labels.tolist()):
        try:
            sequence, _, _ = _parse_crop_path(Path(path))
        except ValueError as error:
            raise ValueError(
                "sample_per_sequence=True reads each crop's sequence from its path, named "
                "{sequence}_{track}/{sequence}_{frame}.jpg as generate_mot_patches writes it, but got "
                f"{path}. Leave sample_per_sequence off for crops named another way."
            ) from error
        sequence_of_identity.setdefault(label, sequence)
    identities_by_sequence: dict[str, list[int]] = {}
    for identity, sequence in sequence_of_identity.items():
        identities_by_sequence.setdefault(sequence, []).append(identity)

    # A batch comes from one sequence and needs 2 identities
    single = sorted(sequence for sequence, identities in identities_by_sequence.items() if len(identities) == 1)
    if len(single) == len(identities_by_sequence):
        raise ValueError(
            "sample_per_sequence=True needs a sequence with at least 2 identities, since every batch comes from "
            "one sequence, but each sequence in train_data holds a single identity."
        )
    if single:
        warnings.warn(
            "Sequences with a single identity will be left out of the batches, since every batch comes from one "
            f"sequence and needs 2 identities: {', '.join(single)}.",
            UserWarning,
            stacklevel=3,
        )
    return [identities for identities in identities_by_sequence.values() if len(identities) >= 2]
