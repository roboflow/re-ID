# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Dataset, PK sampler, and train transforms for ReID fine-tuning."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import TYPE_CHECKING

from PIL import Image
from torch.utils.data import Dataset, Sampler

from reid.data import ReIDSplit
from reid.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    REID_INPUT_SIZE,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    import torch


def build_train_transform(
    input_size: tuple[int, int] = REID_INPUT_SIZE,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
):
    """Training augmentations matching inference geometry (resize, flip, jitter, erase)."""
    from torchvision.transforms import (
        ColorJitter,
        Compose,
        InterpolationMode,
        Normalize,
        RandomErasing,
        RandomHorizontalFlip,
        Resize,
        ToTensor,
    )

    return Compose(
        [
            Resize(input_size, interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.2, contrast=0.15, saturation=0.15, hue=0.0),
            ToTensor(),
            Normalize(mean=list(mean), std=list(std)),
            RandomErasing(p=0.5, value="random"),
        ]
    )


class ReIDTrainDataset(Dataset):
    """Crop dataset over a :class:`ReIDSplit`, with dense training labels.

    The dataset does not touch the filesystem to discover identities; it consumes
    a :class:`ReIDSplit` (typically from :func:`reid.data.load_identity_crops` or
    a holdout partition) and reads crops from ``split.image_paths`` on demand.
    Person IDs are remapped to contiguous ``0..N-1`` classifier labels, so a split
    whose ``pids`` are non-contiguous (for example a train partition left after an
    identity holdout) still yields a dense label space for the classifier head.
    """

    def __init__(self, split: ReIDSplit, transform=None) -> None:
        """Build a training dataset from ``split``.

        Args:
            split: Crop split providing image paths and person IDs.
            transform: Optional image transform (defaults to
                :func:`build_train_transform`).

        Raises:
            ValueError: If ``split`` is empty.
        """
        if len(split) == 0:
            raise ValueError("Cannot build a training dataset from an empty ReIDSplit.")

        self.transform = transform if transform is not None else build_train_transform()
        self.image_paths = list(split.image_paths)

        split_pids = [int(pid) for pid in split.pids.tolist()]
        label_by_pid = {pid: label for label, pid in enumerate(sorted(set(split_pids)))}
        self.labels = [label_by_pid[pid] for pid in split_pids]
        self._num_classes = len(label_by_pid)

    @property
    def num_classes(self) -> int:
        """Number of distinct identities."""
        return self._num_classes

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.image_paths[index]) as img:
            image = img.convert("RGB")
        return self.transform(image), self.labels[index]


class PKSampler(Sampler[list[int]]):
    """Sample ``P`` identities x ``K`` instances per batch for triplet mining."""

    def __init__(
        self,
        labels: list[int],
        p: int,
        k: int,
        seed: int | None = None,
    ) -> None:
        """Initialize the PK sampler.

        Args:
            labels: Per-sample identity labels.
            p: Identities per batch.
            k: Instances per identity per batch.
            seed: Optional shuffle seed.
        """
        if p <= 0 or k <= 0:
            raise ValueError(f"p and k must be positive, got p={p}, k={k}.")

        self.p = p
        self.k = k
        self.batch_size = p * k
        self.seed = seed
        self._epoch = 0

        self._indices_by_label: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            self._indices_by_label[label].append(index)
        self._labels = list(self._indices_by_label)

        if len(self._labels) < p:
            raise ValueError(f"PKSampler needs at least p={p} identities, got {len(self._labels)}.")

        self._num_batches = len(self._labels) // p

    def __len__(self) -> int:
        return self._num_batches

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch so shuffling differs across epochs (deterministically)."""
        self._epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        # Epoch is set by train_reid via set_epoch; do not bump here.
        rng = random.Random(None if self.seed is None else self.seed + self._epoch)  # noqa: S311

        per_label_pool: dict[int, list[int]] = {}
        for label, indices in self._indices_by_label.items():
            shuffled = indices.copy()
            rng.shuffle(shuffled)
            per_label_pool[label] = shuffled

        labels = self._labels.copy()
        rng.shuffle(labels)

        cursor = {label: 0 for label in labels}
        for batch_start in range(self._num_batches):
            chosen = labels[batch_start * self.p : (batch_start + 1) * self.p]
            batch: list[int] = []
            for label in chosen:
                pool = per_label_pool[label]
                picks: list[int] = []
                for _ in range(self.k):
                    if cursor[label] >= len(pool):
                        rng.shuffle(pool)
                        cursor[label] = 0
                    picks.append(pool[cursor[label]])
                    cursor[label] += 1
                batch.extend(picks)
            yield batch
