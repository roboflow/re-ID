# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""The ReID split: image paths with their person and camera IDs."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from reid.checks import validate_positive_int


@dataclass(frozen=True)
class ReIDSplit:
    """Holds one split: its image paths and each image's person and camera IDs.

    A split is a query split, a gallery split, or a full dataset that an
    evaluation protocol divides into query and gallery. Its number of images is
    fixed once built, so its IDs always line up with its images; to add, remove,
    or relabel images, build a new split, which is checked again.

    Attributes:
        image_paths: Path of each image, ``N`` in total, stored as a tuple.
        pids: Person ID of each image. Shape ``(N,)``.
        camids: Camera ID of each image. Shape ``(N,)``.
        gallery_junk_pids: Person IDs whose gallery images are ignored when
            ranking, such as Market-1501's bad detections (``-1``).
    """

    image_paths: Sequence[str]
    pids: np.ndarray
    camids: np.ndarray
    gallery_junk_pids: frozenset[int] = frozenset({-1})

    def __post_init__(self) -> None:
        """Check that each image has one person ID and one camera ID, so no image is ranked as the wrong person."""
        object.__setattr__(self, "image_paths", tuple(self.image_paths))
        expected_shape = (len(self.image_paths),)
        if self.pids.shape != expected_shape:
            raise ValueError(
                f"pids must hold one ID per image: expected shape {expected_shape}, got {self.pids.shape}."
            )
        if self.camids.shape != expected_shape:
            raise ValueError(
                f"camids must hold one ID per image: expected shape {expected_shape}, got {self.camids.shape}."
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def train_val_split(self, num_val_identities: int, seed: int = 0) -> tuple[ReIDSplit, ReIDSplit]:
        """Divide the images by person into a training split and a validation split.

        Every image of ``num_val_identities`` people, drawn at random, goes to the validation split, and the rest stay
        in the training split, so no person is in both. Use it to validate on a benchmark that has no validation
        split, such as Market-1501. Person and camera IDs are kept as they are.

        Args:
            num_val_identities: Number of people in the validation split, at least 1 and fewer than the people in the
                split.
            seed: Seed for drawing the people.

        Returns:
            The training split and the validation split.

        Raises:
            ValueError: If ``num_val_identities`` is below 1, or not fewer than the people in the split.
        """
        validate_positive_int(num_val_identities, "num_val_identities")
        people = np.unique(self.pids).tolist()
        if num_val_identities >= len(people):
            raise ValueError(
                f"num_val_identities must be fewer than the {len(people)} people in the split, "
                f"got {num_val_identities}."
            )
        drawn = np.isin(self.pids, random.Random(seed).sample(people, num_val_identities))
        paths = np.asarray(self.image_paths)
        train = ReIDSplit(
            image_paths=paths[~drawn].tolist(),
            pids=self.pids[~drawn],
            camids=self.camids[~drawn],
            gallery_junk_pids=self.gallery_junk_pids,
        )
        val = ReIDSplit(
            image_paths=paths[drawn].tolist(),
            pids=self.pids[drawn],
            camids=self.camids[drawn],
            gallery_junk_pids=self.gallery_junk_pids,
        )
        return train, val
