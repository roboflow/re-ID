# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""ReID query/gallery split containers and benchmark-specific junk-pid rules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Market-1501 gallery ``0000_*`` images are distractors (pid 0). Query identities
# remain valid when pid=0 on other datasets (e.g. MSMT17).
MARKET1501_GALLERY_JUNK_PIDS = frozenset({-1, 0})


@dataclass
class ReIDSplit:
    """Query or gallery split: image paths, person IDs, and camera IDs.

    ``gallery_junk_pids`` controls which gallery person IDs are excluded from
    ranking during retrieval evaluation.
    """

    image_paths: list[str]
    pids: np.ndarray
    camids: np.ndarray
    gallery_junk_pids: frozenset[int] = frozenset({-1})

    def __post_init__(self) -> None:
        """Check that each image has one person ID and one camera ID, so no image is ranked as the wrong person."""
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
