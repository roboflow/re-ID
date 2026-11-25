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


def test_train_val_split_moves_whole_people_to_val() -> None:
    split = ReIDSplit(
        image_paths=["a1", "a2", "b1", "b2", "c1", "c2"],
        pids=np.array([1, 1, 2, 2, 3, 3]),
        camids=np.array([0, 1, 0, 1, 0, 1]),
    )

    train, val = split.train_val_split(num_val_identities=1)

    person = int(val.pids[0])
    assert list(val.pids) == [person, person]  # both images of the drawn person
    assert list(val.camids) == [0, 1]  # each image keeps its camera
    assert sorted(set(train.pids.tolist())) == sorted({1, 2, 3} - {person})  # everyone else, no one in both
    assert len(train) == 4


def test_train_val_split_skips_junk_images_with_a_warning() -> None:
    split = ReIDSplit(
        image_paths=["bad_detection", "a1", "a2", "b1", "b2"],
        pids=np.array([-1, 1, 1, 2, 2]),  # -1 marks a bad detection, as in Market-1501
        camids=np.array([0, 0, 1, 0, 1]),
    )

    with pytest.warns(UserWarning, match="Skipping 1 of the 5 images, whose person IDs -1 mark junk"):
        train, val = split.train_val_split(num_val_identities=1)

    assert "bad_detection" not in train.image_paths + val.image_paths
    assert len(train) + len(val) == 4
