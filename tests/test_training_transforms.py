# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""``build_train_transform``: augmentation around the inference crop pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import RandomErasing, v2

from reid import ReIDPreprocessing
from reid.training.transforms import build_train_transform


def test_no_augmentations_gives_the_inference_pipeline() -> None:
    preprocessing = ReIDPreprocessing(input_size=(64, 32))
    crop = Image.fromarray(np.random.default_rng(0).integers(0, 256, (80, 40, 3), dtype=np.uint8))

    train_tensor = build_train_transform(preprocessing, ())(crop)

    assert torch.equal(train_tensor, preprocessing.build_crop_transform()(crop))


@pytest.mark.parametrize(
    "erasing_class",
    [pytest.param(RandomErasing, id="transforms"), pytest.param(v2.RandomErasing, id="transforms.v2")],
)
def test_random_erasing_runs_on_the_normalised_tensor(erasing_class: type) -> None:
    preprocessing = ReIDPreprocessing(input_size=(64, 32))
    crop = Image.new("RGB", (32, 64), (255, 255, 255))  # white normalises to about 2, never to 0

    tensor = build_train_transform(preprocessing, [erasing_class(p=1.0, value=0)])(crop)

    assert (tensor == 0).any()
