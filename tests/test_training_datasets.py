# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for the ReID train dataset and the PK sampler."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from reid.data import ReIDSplit, load_identity_crops
from reid.preprocessing import ReIDPreprocessing
from reid.training.datasets import (
    PKSampler,
    ReIDTrainDataset,
    build_eval_transform,
    build_train_transform,
)


def _make_crops(root: Path, num_ids: int = 4, num_per: int = 6) -> None:
    rng = np.random.default_rng(0)
    for i in range(num_ids):
        identity_dir = root / f"id_{i:02d}"
        identity_dir.mkdir(parents=True)
        for j in range(num_per):
            array = (rng.random((48, 24, 3)) * 255).astype("uint8")
            Image.fromarray(array).save(identity_dir / f"{j}.jpg")


class TestTransforms:
    """Training transforms must resize exactly like inference."""

    def test_eval_transform_matches_inference_resize(self) -> None:
        """Training-side resize honours ``resize_mode`` / ``interpolation`` exactly like ``resize_crop``."""
        preprocessing = ReIDPreprocessing(resize_mode="letterbox", interpolation="bicubic")
        crop = (np.random.default_rng(0).random((200, 50, 3)) * 255).astype("uint8")

        tensor = build_eval_transform(preprocessing)(Image.fromarray(crop))
        pixels = (
            tensor * torch.tensor(preprocessing.std)[:, None, None] + torch.tensor(preprocessing.mean)[:, None, None]
        ) * 255

        expected = torch.from_numpy(preprocessing.resize_crop(crop)).permute(2, 0, 1).float()
        assert torch.allclose(pixels, expected, atol=0.5)
        # A 200x50 crop letterboxed into 256x128 scales to 256x64, so columns 64+ are padding.
        assert torch.all(pixels[:, :, 64:] == preprocessing.pad_value)


class TestReIDTrainDataset:
    """Crop dataset construction and dense label remapping."""

    def test_reads_split_and_labels(self, tmp_path: Path) -> None:
        _make_crops(tmp_path, num_ids=4, num_per=5)
        dataset = ReIDTrainDataset(load_identity_crops(tmp_path), transform=build_train_transform())

        assert dataset.num_classes == 4
        assert len(dataset) == 20
        image, label = dataset[0]
        assert tuple(image.shape) == (3, 256, 128)
        assert 0 <= label < 4

    def test_remaps_noncontiguous_pids_to_dense_labels(self) -> None:
        split = ReIDSplit(
            image_paths=["a.jpg", "b.jpg", "c.jpg"],
            pids=np.array([2, 5, 5], dtype=np.int32),
            camids=np.array([-1, -1, -1], dtype=np.int32),
        )
        dataset = ReIDTrainDataset(split, transform=lambda image: image)

        assert dataset.num_classes == 2
        assert dataset.labels == [0, 1, 1]

    def test_from_identity_label_map_keeps_overlapping_folders(self, tmp_path: Path) -> None:
        train_root = tmp_path / "train"
        val_root = tmp_path / "val"
        _make_crops(train_root, num_ids=4, num_per=3)
        for name in ("id_00", "id_02"):
            src = train_root / name
            dst = val_root / name
            dst.mkdir(parents=True)
            for path in src.glob("*.jpg"):
                Image.open(path).save(dst / path.name)
        unseen = val_root / "id_99"
        unseen.mkdir(parents=True)
        Image.fromarray(np.zeros((48, 24, 3), dtype="uint8")).save(unseen / "0.jpg")

        train_ds = ReIDTrainDataset(load_identity_crops(train_root), transform=lambda image: image)
        val_ds = ReIDTrainDataset.from_identity_label_map(
            load_identity_crops(val_root),
            train_ds.label_by_identity,
            transform=lambda image: image,
            num_classes=train_ds.num_classes,
        )

        assert len(val_ds) == 6
        assert set(val_ds.labels) == {
            train_ds.label_by_identity["id_00"],
            train_ds.label_by_identity["id_02"],
        }
        assert val_ds.num_classes == 4


class TestPKSampler:
    """PK batch structure and epoch budgets."""

    def test_batch_structure(self, tmp_path: Path) -> None:
        _make_crops(tmp_path, num_ids=5, num_per=4)
        dataset = ReIDTrainDataset(load_identity_crops(tmp_path))
        p, k = 2, 3
        sampler = PKSampler(dataset.labels, p=p, k=k, seed=123)

        batches = list(sampler)
        assert len(sampler) == 5 // p
        assert len(batches) == len(sampler)

        for batch in batches:
            assert len(batch) == p * k
            label_counts = Counter(dataset.labels[i] for i in batch)
            assert len(label_counts) == p
            assert all(count == k for count in label_counts.values())

    def test_epoch_budget_scales_with_identities_and_crops(self, tmp_path: Path) -> None:
        """Batches per epoch are ``(identities_per_epoch * crops_per_identity) // (p * k)``."""
        num_ids = 8
        _make_crops(tmp_path, num_ids=num_ids, num_per=10)
        dataset = ReIDTrainDataset(load_identity_crops(tmp_path))
        p, k = 2, 2

        default = PKSampler(dataset.labels, p=p, k=k, seed=0)
        deeper = PKSampler(dataset.labels, p=p, k=k, seed=0, crops_per_identity=2 * k)
        shorter = PKSampler(dataset.labels, p=p, k=k, seed=0, identities_per_epoch=num_ids // 2)

        # len(sampler) is the epoch's batch count. crops_per_identity defaults to k,
        # so k cancels and the default epoch is one pass over the identities.
        assert len(default) == num_ids // p
        assert len(deeper) == 2 * len(default)
        assert len(shorter) == len(default) // 2

    def test_rejects_too_few_identities(self, tmp_path: Path) -> None:
        _make_crops(tmp_path, num_ids=2, num_per=4)
        dataset = ReIDTrainDataset(load_identity_crops(tmp_path))
        with pytest.raises(ValueError, match="at least p="):
            PKSampler(dataset.labels, p=4, k=2)
