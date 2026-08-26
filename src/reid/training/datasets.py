# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Dataset, PK sampler, and train transforms for ReID fine-tuning."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Literal

from torch.utils.data import Sampler
from torchvision.transforms import (
    ColorJitter,
    Compose,
    RandomErasing,
    RandomHorizontalFlip,
)

from reid.crop_dataset import CropDataset
from reid.data import (
    ReIDSplit,
    dense_labels_from_split,
    identity_name_from_crop_path,
    indices_by_label,
    labels_from_identity_map,
)
from reid.preprocessing import ReIDPreprocessing

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import torch
    from PIL import Image

IdentitiesPerEpoch = int | Literal["all"]


def build_train_transform(preprocessing: ReIDPreprocessing | None = None) -> Compose:
    """Training augmentations matching inference geometry (resize, flip, jitter, erase)."""
    preprocessing = preprocessing or ReIDPreprocessing()
    return Compose(
        [
            preprocessing.resize_transform(),
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.2, contrast=0.15, saturation=0.15, hue=0.0),
            *preprocessing.build_transform().transforms,
            RandomErasing(p=0.5, value="random"),
        ]
    )


def build_eval_transform(preprocessing: ReIDPreprocessing | None = None) -> Compose:
    """Deterministic resize + normalize for validation (no augmentation)."""
    preprocessing = preprocessing or ReIDPreprocessing()
    return preprocessing.build_crop_transform()


class ReIDTrainDataset(CropDataset):
    """Crop dataset yielding ``(tensor, label)`` for classifier training.

    Adds the training label space to :class:`~reid.crop_dataset.CropDataset`:
    dense ``0..N-1`` classifier labels, the identity-folder to label map that
    validation reuses, and per-item labels for :class:`PKSampler`. Build it with
    :meth:`from_split` (remaps a split's person IDs) or
    :meth:`from_identity_label_map` (reuses a training map for val crops).
    """

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        *,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        num_classes: int | None = None,
    ) -> None:
        """Build a training dataset from crop paths and dense classifier labels.

        Args:
            image_paths: Crop paths.
            labels: Classifier labels aligned with ``image_paths``.
            transform: Image transform (defaults to :func:`build_train_transform`).
            num_classes: Classifier class count. Defaults to ``max(labels) + 1``.

        Raises:
            ValueError: If the inputs are empty or of unequal length.
        """
        if not image_paths:
            raise ValueError("Cannot build a training dataset from empty image_paths.")
        if len(image_paths) != len(labels):
            raise ValueError("image_paths and labels must have the same length.")

        super().__init__(image_paths, transform if transform is not None else build_train_transform())
        self.labels = [int(label) for label in labels]
        self._num_classes = num_classes if num_classes is not None else max(self.labels) + 1
        self._label_by_identity = {
            identity_name_from_crop_path(path): label for path, label in zip(self.image_paths, self.labels, strict=True)
        }

    @classmethod
    def from_split(
        cls,
        split: ReIDSplit,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> ReIDTrainDataset:
        """Build from a :class:`ReIDSplit`, remapping person IDs to dense labels."""
        image_paths, labels, num_classes = dense_labels_from_split(split)
        return cls(image_paths, labels, transform=transform, num_classes=num_classes)

    @classmethod
    def from_identity_label_map(
        cls,
        split: ReIDSplit,
        label_by_identity: dict[str, int],
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        *,
        num_classes: int | None = None,
    ) -> ReIDTrainDataset:
        """Build val crops that reuse the training identity folder → label map."""
        image_paths, labels = labels_from_identity_map(split, label_by_identity)
        return cls(
            image_paths,
            labels,
            transform=transform,
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return super().__getitem__(index), self.labels[index]


def _resolve_pk_num_batches(
    num_identities: int,
    p: int,
    k: int,
    *,
    identities_per_epoch: IdentitiesPerEpoch = "all",
    crops_per_identity: int,
) -> int:
    """Return how many PK batches one epoch should contain.

    Budget is ``identities_per_epoch * crops_per_identity`` crop draws, packed into
    batches of size ``p * k``.
    """
    if p <= 0 or k <= 0:
        raise ValueError(f"p and k must be positive, got p={p}, k={k}.")
    if crops_per_identity < 1:
        raise ValueError(f"crops_per_identity must be >= 1, got {crops_per_identity}.")
    n_identities = num_identities if identities_per_epoch == "all" else identities_per_epoch
    if not isinstance(n_identities, int) or n_identities < 1:
        raise ValueError(f"identities_per_epoch must be 'all' or an int >= 1, got {identities_per_epoch!r}.")
    if n_identities < p:
        raise ValueError(
            f"identities_per_epoch must be >= p={p} (got {n_identities} "
            f"from identities_per_epoch={identities_per_epoch!r}, "
            f"num_identities={num_identities})."
        )
    return max(1, (n_identities * crops_per_identity) // (p * k))


class PKSampler(Sampler[list[int]]):
    """Sample ``P`` identities x ``K`` instances per batch for triplet mining.

    Epoch length is an identity/crop budget, not a full pass over every frame:

    ``batches = (identities_per_epoch * crops_per_identity) // (p * k)``

    With defaults (``identities_per_epoch="all"``, ``crops_per_identity=k``) this
    is ``num_identities // p``. Identities are reshuffled within the epoch when
    a pass over the ID list is exhausted.
    """

    def __init__(
        self,
        labels: list[int],
        p: int,
        k: int,
        seed: int | None = None,
        *,
        identities_per_epoch: IdentitiesPerEpoch = "all",
        crops_per_identity: int | None = None,
    ) -> None:
        """Initialize the PK sampler.

        Args:
            labels: Per-sample identity labels.
            p: Identities per batch.
            k: Instances per identity per batch.
            seed: Optional shuffle seed.
            identities_per_epoch: Identity slots to cover per epoch (``"all"``
                uses every label in ``labels``).
            crops_per_identity: Crop draws per identity slot per epoch. ``None``
                uses ``k``.
        """
        if p <= 0 or k <= 0:
            raise ValueError(f"p and k must be positive, got p={p}, k={k}.")

        self.p = p
        self.k = k
        self.batch_size = p * k
        self.seed = seed
        self.identities_per_epoch = identities_per_epoch
        self.crops_per_identity = k if crops_per_identity is None else crops_per_identity
        self._epoch = 0

        self._indices_by_label = indices_by_label(labels)
        self._labels = list(self._indices_by_label)

        if len(self._labels) < p:
            raise ValueError(f"PKSampler needs at least p={p} identities, got {len(self._labels)}.")

        self._num_batches = _resolve_pk_num_batches(
            num_identities=len(self._labels),
            p=p,
            k=k,
            identities_per_epoch=identities_per_epoch,
            crops_per_identity=self.crops_per_identity,
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
