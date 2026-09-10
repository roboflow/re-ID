# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Crop reading shared by inference and training."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from PIL import Image as PILImage
from torch.utils.data import Dataset


class CropDataset(Dataset):
    """Reads crop images from disk and applies ``transform`` on worker processes.

        Internal building block shared by inference and training; not part of the
        curated public API surface.

        Crops on disk are read as RGB, so the transform owns every pixel step
    (resize, augmentation, tensor conversion). Inference and training share
        this reader; training adds labels in
        :class:`~reid.training.datasets.ReIDTrainDataset`.

        Internal building block; not part of the curated public API surface.
    """

    def __init__(self, image_paths: Sequence[str], transform: Callable[[PILImage.Image], torch.Tensor]) -> None:
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with PILImage.open(self.image_paths[index]) as img:
            return self.transform(img.convert("RGB"))
