# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Dataset, PK sampler, and train transforms for ReID fine-tuning."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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

# How many PK batches make one epoch:
# - identities: num_identities // p (short; current default)
# - samples: max(1, num_samples // (p * k)) (FastReID-style step budget)
EpochLength = Literal["identities", "samples"]
_EPOCH_LENGTH_CHOICES: tuple[str, ...] = ("identities", "samples")


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


def build_eval_transform(
    input_size: tuple[int, int] = REID_INPUT_SIZE,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
):
    """Deterministic resize + normalize for validation (no augmentation)."""
    from torchvision.transforms import (
        Compose,
        InterpolationMode,
        Normalize,
        Resize,
        ToTensor,
    )

    return Compose(
        [
            Resize(input_size, interpolation=InterpolationMode.BILINEAR),
            ToTensor(),
            Normalize(mean=list(mean), std=list(std)),
        ]
    )


def identity_name_from_crop_path(path: str | Path) -> str:
    """Identity folder name for ``<root>/<identity>/<crop>.jpg`` layouts."""
    return Path(path).parent.name


class ReIDTrainDataset(Dataset):
    """Crop dataset over a :class:`ReIDSplit`, with dense training labels.

    The dataset does not touch the filesystem to discover identities; it consumes
    a :class:`ReIDSplit` (typically from :func:`reid.data.load_identity_crops` or
    a holdout partition) and reads crops from ``split.image_paths`` on demand.
    Person IDs are remapped to contiguous ``0..N-1`` classifier labels, so a split
    whose ``pids`` are non-contiguous (for example a train partition left after an
    identity holdout) still yields a dense label space for the classifier head.
    """

    def __init__(
        self,
        split: ReIDSplit | None = None,
        transform=None,
        *,
        image_paths: list[str] | None = None,
        labels: list[int] | None = None,
        num_classes: int | None = None,
    ) -> None:
        """Build a training dataset from ``split`` or explicit paths/labels.

        Args:
            split: Crop split providing image paths and person IDs.
            transform: Optional image transform (defaults to
                :func:`build_train_transform`).
            image_paths: Explicit crop paths (with ``labels``).
            labels: Explicit classifier labels aligned with ``image_paths``.
            num_classes: Classifier class count when using explicit labels.
                Defaults to ``max(labels) + 1``.

        Raises:
            ValueError: If inputs are empty or inconsistent.
        """
        if split is not None:
            if image_paths is not None or labels is not None:
                raise ValueError("Pass either split or image_paths/labels, not both.")
            if len(split) == 0:
                raise ValueError("Cannot build a training dataset from an empty ReIDSplit.")

            self.image_paths = list(split.image_paths)
            split_pids = [int(pid) for pid in split.pids.tolist()]
            label_by_pid = {pid: label for label, pid in enumerate(sorted(set(split_pids)))}
            self.labels = [label_by_pid[pid] for pid in split_pids]
            self._num_classes = len(label_by_pid)
        else:
            if image_paths is None or labels is None:
                raise ValueError("Provide split, or both image_paths and labels.")
            if len(image_paths) == 0:
                raise ValueError("Cannot build a training dataset from empty image_paths.")
            if len(image_paths) != len(labels):
                raise ValueError("image_paths and labels must have the same length.")
            self.image_paths = list(image_paths)
            self.labels = [int(label) for label in labels]
            self._num_classes = num_classes if num_classes is not None else max(self.labels) + 1

        self.transform = transform if transform is not None else build_train_transform()
        self._label_by_identity = {
            identity_name_from_crop_path(path): label for path, label in zip(self.image_paths, self.labels, strict=True)
        }

    @classmethod
    def from_identity_label_map(
        cls,
        split: ReIDSplit,
        label_by_identity: dict[str, int],
        transform=None,
        *,
        num_classes: int | None = None,
    ) -> ReIDTrainDataset:
        """Keep crops whose identity folder is in ``label_by_identity``.

        Used for temporal val crops that reuse training identity folder names
        (for example MOT17 train-half / val-half of the same sequences).
        """
        image_paths: list[str] = []
        labels: list[int] = []
        for path in split.image_paths:
            name = identity_name_from_crop_path(path)
            label = label_by_identity.get(name)
            if label is None:
                continue
            image_paths.append(path)
            labels.append(label)
        if not image_paths:
            raise ValueError("No validation crops share identity folder names with the training map.")
        return cls(
            transform=transform,
            image_paths=image_paths,
            labels=labels,
            num_classes=num_classes if num_classes is not None else max(label_by_identity.values()) + 1,
        )

    @property
    def num_classes(self) -> int:
        """Number of distinct identities."""
        return self._num_classes

    @property
    def label_by_identity(self) -> dict[str, int]:
        """Map identity folder name → classifier label."""
        return dict(self._label_by_identity)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.image_paths[index]) as img:
            image = img.convert("RGB")
        return self.transform(image), self.labels[index]


def _resolve_pk_num_batches(
    num_samples: int,
    num_identities: int,
    p: int,
    k: int,
    epoch_length: EpochLength,
) -> int:
    """Return how many PK batches one epoch should contain."""
    if epoch_length == "identities":
        return num_identities // p
    if epoch_length == "samples":
        return max(1, num_samples // (p * k))
    choices = ", ".join(repr(choice) for choice in _EPOCH_LENGTH_CHOICES)
    raise ValueError(f"Unknown epoch_length {epoch_length!r}. Choose from: [{choices}].")


class PKSampler(Sampler[list[int]]):
    """Sample ``P`` identities x ``K`` instances per batch for triplet mining.

    ``epoch_length`` controls how many batches make one epoch:

    - ``"identities"``: ``num_identities // p`` (default; short when many images
      per identity).
    - ``"samples"``: ``max(1, num_samples // (p * k))`` — a FastReID-style step
      budget ≈ one pass over the dataset in PK batch units. Identities are
      reshuffled within the epoch when a pass over the ID list is exhausted.
    """

    def __init__(
        self,
        labels: list[int],
        p: int,
        k: int,
        seed: int | None = None,
        *,
        epoch_length: EpochLength = "identities",
    ) -> None:
        """Initialize the PK sampler.

        Args:
            labels: Per-sample identity labels.
            p: Identities per batch.
            k: Instances per identity per batch.
            seed: Optional shuffle seed.
            epoch_length: How many batches constitute one epoch.
        """
        if p <= 0 or k <= 0:
            raise ValueError(f"p and k must be positive, got p={p}, k={k}.")

        self.p = p
        self.k = k
        self.batch_size = p * k
        self.seed = seed
        self.epoch_length = epoch_length
        self._epoch = 0

        self._indices_by_label: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            self._indices_by_label[label].append(index)
        self._labels = list(self._indices_by_label)

        if len(self._labels) < p:
            raise ValueError(f"PKSampler needs at least p={p} identities, got {len(self._labels)}.")

        self._num_batches = _resolve_pk_num_batches(
            num_samples=len(labels),
            num_identities=len(self._labels),
            p=p,
            k=k,
            epoch_length=epoch_length,
        )

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
        label_cursor = 0
        index_cursor = {label: 0 for label in labels}

        for _ in range(self._num_batches):
            if label_cursor + self.p > len(labels):
                rng.shuffle(labels)
                label_cursor = 0
            chosen = labels[label_cursor : label_cursor + self.p]
            label_cursor += self.p

            batch: list[int] = []
            for label in chosen:
                pool = per_label_pool[label]
                picks: list[int] = []
                for _ in range(self.k):
                    if index_cursor[label] >= len(pool):
                        rng.shuffle(pool)
                        index_cursor[label] = 0
                    picks.append(pool[index_cursor[label]])
                    index_cursor[label] += 1
                batch.extend(picks)
            yield batch
