# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Training augmentation around the inference crop pipeline."""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image
from torch import nn
from torchvision.transforms import Compose, RandomErasing, v2

from reid.preprocessing import ReIDPreprocessing


def build_train_transform(preprocessing: ReIDPreprocessing, augmentations: Sequence[nn.Module]) -> Compose:
    """Build the training pipeline for an RGB PIL crop.

    The pipeline runs the model's resize, the augmentations in order, then the model's ``ToTensor`` and ``Normalize``.
    ``RandomErasing`` (``torchvision.transforms`` or ``transforms.v2``) works on tensors, so it runs after
    ``Normalize``.

    Args:
        preprocessing: The model's preprocessing, which sets the resize and the normalisation.
        augmentations: ``torchvision.transforms`` objects. ``()`` gives the inference pipeline.

    Returns:
        Transform from an RGB PIL image to a normalised tensor at ``preprocessing.input_size``.

    Raises:
        ValueError: If the augmentations change the crop's size.
    """
    # RandomErasing works on tensors, so it runs after Normalize; every other augmentation runs on the image
    on_image, on_tensor = [], []
    for augmentation in augmentations:
        if isinstance(augmentation, (RandomErasing, v2.RandomErasing)):
            on_tensor.append(augmentation)
        else:
            on_image.append(augmentation)
    resize, to_tensor, normalize = preprocessing.build_crop_transform().transforms
    pipeline = Compose([resize, *on_image, to_tensor, normalize, *on_tensor])

    height, width = preprocessing.input_size
    _, crop_height, crop_width = pipeline(Image.new("RGB", (width, height))).shape
    if (crop_height, crop_width) != (height, width):
        raise ValueError(
            f"augmentations give {crop_height}x{crop_width} crops, but the model's input size is {height}x{width}, "
            "so it would train on crops of a size inference never sees. Leave out the transforms that crop or "
            "resize, or change input_size in the model's preprocessing."
        )
    return pipeline
