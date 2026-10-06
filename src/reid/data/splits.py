# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""The ReID split: image paths with their person and camera IDs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


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
