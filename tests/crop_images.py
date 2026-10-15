# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Write crop images to disk in the folder layouts the library reads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _write_image(path: Path, rng: np.random.Generator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rng.random((48, 24, 3)) * 255).astype("uint8")).save(path)


def write_identity_crops(root: Path, num_ids: int, num_per: int, *, prefix: str = "id") -> None:
    """Write random crops to ``<root>/<prefix>_<id>/<index>.jpg``."""
    rng = np.random.default_rng(0)
    for identity in range(num_ids):
        for index in range(num_per):
            _write_image(root / f"{prefix}_{identity:02d}" / f"{index:02d}.jpg", rng)


def write_mot_crop(root: Path, sequence: str, track_id: int, frame: int) -> Path:
    """Write one random crop named like ``generate_mot_patches`` output."""
    path = root / f"{sequence}_{track_id}" / f"{sequence}_{frame:06d}.jpg"
    _write_image(path, np.random.default_rng(track_id * 100_000 + frame))
    return path
