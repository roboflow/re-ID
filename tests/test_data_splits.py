# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""``ReIDSplit``: image paths with their person and camera IDs."""

from __future__ import annotations

import numpy as np
import pytest

from reid.data import ReIDSplit


@pytest.mark.parametrize("field", ["pids", "camids"])
def test_ids_must_line_up_with_the_images(field: str) -> None:
    ids = {"pids": np.array([1]), "camids": np.array([0])}
    ids[field] = np.array([1, 2])  # two IDs for one image

    with pytest.raises(ValueError, match=rf"{field} must hold one ID per image: expected shape \(1,\), got \(2,\)"):
        ReIDSplit(image_paths=["q"], **ids)


def test_images_cannot_be_added_after_the_check() -> None:
    split = ReIDSplit(image_paths=["q"], pids=np.array([1]), camids=np.array([0]))

    with pytest.raises(AttributeError):
        split.image_paths.append("extra")  # type: ignore[attr-defined]

    assert len(split.image_paths) == len(split.pids) == 1
