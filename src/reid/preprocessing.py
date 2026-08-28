# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""ReID crop and embedding preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import cv2
import numpy as np
from PIL import Image as PILImage
from torchvision.transforms import Compose, Normalize, ToTensor

# ImageNet statistics — shared by every OSNet / timm ImageNet-pretrained backbone.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Standard person re-ID input geometry (height, width). OSNet and most backbones
# train at 256x128.
REID_INPUT_SIZE = (256, 128)

# Gray padding used by letterbox resize.
LETTERBOX_PAD_VALUE = 114

ResizeMode = Literal["stretch", "letterbox"]


class Interpolation(str, Enum):
    """Supported crop interpolation modes."""

    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    NEAREST = "nearest"


_OPENCV_INTERPOLATION = {
    Interpolation.BILINEAR: cv2.INTER_LINEAR,
    Interpolation.BICUBIC: cv2.INTER_CUBIC,
    Interpolation.NEAREST: cv2.INTER_NEAREST,
}


@dataclass(frozen=True)
class ReIDPreprocessing:
    """Crop resize and colour/tensor conversion for backbone input.

    Preprocessing does not own embedding L2 normalisation: ``extract_*`` always
    returns raw embeddings and cosine similarity is normalised at distance time
    (see ``ReIDEvaluator._distance_matrix``).
    """

    input_size: tuple[int, int] = REID_INPUT_SIZE
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    interpolation: Interpolation = Interpolation.BILINEAR
    to_rgb: bool = True
    resize_mode: ResizeMode = "stretch"
    pad_value: int = LETTERBOX_PAD_VALUE

    def __post_init__(self) -> None:
        """Normalize interpolation and reject unknown ``resize_mode`` values."""
        try:
            interpolation = Interpolation(self.interpolation)
        except ValueError as error:
            choices = sorted(mode.value for mode in Interpolation)
            raise ValueError(f"Unknown interpolation {self.interpolation!r}. Choose from: {choices}") from error
        object.__setattr__(self, "interpolation", interpolation)

        if self.resize_mode not in ("stretch", "letterbox"):
            raise ValueError(f"Unknown resize_mode {self.resize_mode!r}. Choose from: ['letterbox', 'stretch'].")

    def describe(self) -> str:
        """Human-readable one-line summary."""
        h, w = self.input_size
        colour = "BGR→RGB" if self.to_rgb else "RGB (no swap)"
        return (
            f"ReIDPreprocessing(resize={h}x{w} [{self.resize_mode}, {self.interpolation.value}], "
            f"{colour}, mean={self.mean}, std={self.std})"
        )

    def prepare_crop_array(self, crop: np.ndarray, *, from_bgr_frame: bool) -> np.ndarray:
        """Normalise crop colour to RGB before :meth:`resize_crop`.

        Video frames from OpenCV / supervision are BGR; when ``to_rgb`` is
        ``True``, they are converted here. On-disk crops loaded via PIL are
        already RGB and must not be channel-swapped, regardless of ``to_rgb``.
        """
        if from_bgr_frame and self.to_rgb:
            return crop[:, :, ::-1].copy()
        return crop

    def resize_crop(self, crop: np.ndarray) -> np.ndarray:
        """Resize an ``HxWxC`` crop to ``input_size`` using OpenCV.

        ``stretch`` ignores aspect ratio (``cv2.resize`` to the target size).
        ``letterbox`` preserves aspect ratio and pads with ``pad_value``.

        Returns:
            Crop resized to ``input_size``.
        """
        if crop.size == 0:
            target_h, target_w = self.input_size
            channels = crop.shape[2] if crop.ndim == 3 else 1
            shape = (target_h, target_w, channels) if crop.ndim == 3 else (target_h, target_w)
            return np.zeros(shape, dtype=crop.dtype)

        target_h, target_w = self.input_size
        interpolation = _OPENCV_INTERPOLATION[self.interpolation]

        if self.resize_mode == "letterbox":
            height, width = crop.shape[:2]
            scale = min(target_h / height, target_w / width)
            new_w = max(round(width * scale), 1)
            new_h = max(round(height * scale), 1)
            resized = cv2.resize(crop, (new_w, new_h), interpolation=interpolation)
            if crop.ndim == 3:
                padded = np.full((target_h, target_w, crop.shape[2]), self.pad_value, dtype=crop.dtype)
            else:
                padded = np.full((target_h, target_w), self.pad_value, dtype=crop.dtype)
            padded[:new_h, :new_w] = resized
            return padded

        return cv2.resize(crop, (target_w, target_h), interpolation=interpolation)

    def resize_transform(self) -> ResizeCrop:
        """Transform applying :meth:`resize_crop` to an RGB PIL crop."""
        return ResizeCrop(self)

    def build_crop_transform(self) -> Compose:
        """Full crop pipeline for an RGB PIL image: resize then tensor normalisation."""
        return Compose([self.resize_transform(), *self.build_transform().transforms])

    def build_transform(self) -> Compose:
        """Tensor normalisation for a crop already at ``input_size`` (RGB PIL in)."""
        return Compose(
            [
                ToTensor(),
                Normalize(mean=list(self.mean), std=list(self.std)),
            ]
        )

    def to_dict(self) -> dict:
        """Serialise for ``reid_config.json``."""
        return {
            "input_size": list(self.input_size),
            "mean": list(self.mean),
            "std": list(self.std),
            "interpolation": self.interpolation.value,
            "to_rgb": self.to_rgb,
            "resize_mode": self.resize_mode,
            "pad_value": self.pad_value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ReIDPreprocessing:
        """Reconstruct from a ``reid_config.json`` preprocessing dict.

        Unknown keys are ignored, so configs written by older versions (e.g. a
        legacy ``normalize_embeddings`` flag) still load cleanly.
        """
        kwargs: dict = {}
        if "input_size" in data:
            kwargs["input_size"] = tuple(data["input_size"])
        if "mean" in data:
            kwargs["mean"] = tuple(data["mean"])
        if "std" in data:
            kwargs["std"] = tuple(data["std"])
        if "interpolation" in data:
            kwargs["interpolation"] = data["interpolation"]
        if "to_rgb" in data:
            kwargs["to_rgb"] = data["to_rgb"]
        if "resize_mode" in data:
            kwargs["resize_mode"] = data["resize_mode"]
        if "pad_value" in data:
            kwargs["pad_value"] = data["pad_value"]
        return cls(**kwargs)


class ResizeCrop:
    """Resize an RGB PIL crop with :meth:`ReIDPreprocessing.resize_crop`.

    Keeps ``resize_mode`` / ``interpolation`` and the OpenCV resampler identical
    between training and serving.
    """

    def __init__(self, preprocessing: ReIDPreprocessing) -> None:
        self.preprocessing = preprocessing

    def __call__(self, image: PILImage.Image) -> PILImage.Image:
        return PILImage.fromarray(self.preprocessing.resize_crop(np.asarray(image)))
