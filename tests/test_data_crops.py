# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for identity-crop loading and split partitioning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from reid.data import (
    ReIDSplit,
    build_crop_identity_holdout_split,
    build_crop_retrieval_split,
    load_identity_crops,
)


def _make_crops(root: Path, num_ids: int, num_per: int, *, prefix: str = "id") -> None:
    rng = np.random.default_rng(0)
    for i in range(num_ids):
        identity_dir = root / f"{prefix}_{i:02d}"
        identity_dir.mkdir(parents=True)
        for j in range(num_per):
            array = (rng.random((48, 24, 3)) * 255).astype("uint8")
            Image.fromarray(array).save(identity_dir / f"{j:02d}.jpg")


# --------------------------------------------------------------------------- #
# load_identity_crops
# --------------------------------------------------------------------------- #


def test_load_identity_crops_assigns_contiguous_pids_and_neg_camids(tmp_path: Path) -> None:
    _make_crops(tmp_path, num_ids=3, num_per=4)
    split = load_identity_crops(tmp_path)

    assert isinstance(split, ReIDSplit)
    assert len(split) == 12
    assert sorted(set(split.pids.tolist())) == [0, 1, 2]
    assert set(split.camids.tolist()) == {-1}
    # Sorted identity order -> contiguous, monotonic pids.
    assert split.pids.tolist() == sorted(split.pids.tolist())


def test_load_identity_crops_honours_identity_whitelist(tmp_path: Path) -> None:
    _make_crops(tmp_path, num_ids=4, num_per=3)
    split = load_identity_crops(tmp_path, identities=["id_00", "id_02"])

    assert sorted(set(split.pids.tolist())) == [0, 1]
    assert len(split) == 6
    assert all(("id_00" in p or "id_02" in p) for p in split.image_paths)


def test_load_identity_crops_skips_empty_folders(tmp_path: Path) -> None:
    _make_crops(tmp_path, num_ids=2, num_per=3)
    (tmp_path / "id_empty").mkdir()
    split = load_identity_crops(tmp_path)

    assert sorted(set(split.pids.tolist())) == [0, 1]


def test_load_identity_crops_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Crop dataset root not found"):
        load_identity_crops(tmp_path / "nope")


def test_load_identity_crops_no_crops_raises(tmp_path: Path) -> None:
    (tmp_path / "id_empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No identity crops found"):
        load_identity_crops(tmp_path)


# --------------------------------------------------------------------------- #
# build_crop_retrieval_split
# --------------------------------------------------------------------------- #


def test_retrieval_split_uses_all_eligible_identities(tmp_path: Path) -> None:
    _make_crops(tmp_path, num_ids=8, num_per=4)
    query, gallery = build_crop_retrieval_split(load_identity_crops(tmp_path), queries_per_id=1)

    assert len(query) == 8
    assert len(gallery) == 8 * 3
    assert set(query.camids.tolist()) == {0}
    assert set(gallery.camids.tolist()) == {1}
    # Every query identity is retrievable from the gallery.
    assert set(query.pids.tolist()) == set(gallery.pids.tolist())


def test_retrieval_split_filters_ineligible_identities(tmp_path: Path) -> None:
    _make_crops(tmp_path, num_ids=3, num_per=4)
    (tmp_path / "id_single").mkdir()
    Image.fromarray((np.random.default_rng(1).random((48, 24, 3)) * 255).astype("uint8")).save(
        tmp_path / "id_single" / "00.jpg"
    )
    split = load_identity_crops(tmp_path)

    query, gallery = build_crop_retrieval_split(split, queries_per_id=1, min_crops=2)
    # The single-crop identity is dropped; three eligible identities remain.
    assert len(query) == 3


def test_retrieval_split_is_deterministic(tmp_path: Path) -> None:
    _make_crops(tmp_path, num_ids=5, num_per=4)
    split = load_identity_crops(tmp_path)

    first = build_crop_retrieval_split(split, seed=7)
    second = build_crop_retrieval_split(split, seed=7)
    assert first[0].image_paths == second[0].image_paths
    assert first[1].image_paths == second[1].image_paths


def test_retrieval_split_rejects_incompatible_params(tmp_path: Path) -> None:
    _make_crops(tmp_path, num_ids=3, num_per=4)
    split = load_identity_crops(tmp_path)

    with pytest.raises(ValueError, match="min_crops must be > queries_per_id"):
        build_crop_retrieval_split(split, queries_per_id=2, min_crops=2)
    with pytest.raises(ValueError, match="queries_per_id must be >= 1"):
        build_crop_retrieval_split(split, queries_per_id=0)


def test_retrieval_split_no_eligible_identities_raises(tmp_path: Path) -> None:
    _make_crops(tmp_path, num_ids=3, num_per=2)
    split = load_identity_crops(tmp_path)

    with pytest.raises(ValueError, match="No identity has >= 3 crops"):
        build_crop_retrieval_split(split, min_crops=3)


# --------------------------------------------------------------------------- #
# build_crop_identity_holdout_split
# --------------------------------------------------------------------------- #


def test_holdout_is_identity_disjoint_from_train(tmp_path: Path) -> None:
    _make_crops(tmp_path, num_ids=10, num_per=4)
    train, query, gallery = build_crop_identity_holdout_split(
        load_identity_crops(tmp_path), holdout_fraction=0.2, seed=0
    )

    holdout_pids = set(query.pids.tolist())
    assert set(train.pids.tolist()).isdisjoint(holdout_pids)
    assert len(holdout_pids) == 2
    assert set(query.camids.tolist()) == {0}
    assert set(gallery.camids.tolist()) == {1}
    assert len(query) == len(holdout_pids)
    assert set(query.pids.tolist()) == set(gallery.pids.tolist())


def test_holdout_keeps_ineligible_identities_in_train(tmp_path: Path) -> None:
    _make_crops(tmp_path, num_ids=4, num_per=4)
    (tmp_path / "id_single").mkdir()
    Image.fromarray((np.random.default_rng(2).random((48, 24, 3)) * 255).astype("uint8")).save(
        tmp_path / "id_single" / "00.jpg"
    )
    split = load_identity_crops(tmp_path)

    train, query, _ = build_crop_identity_holdout_split(split, holdout_fraction=1.0, min_crops=2, seed=0)
    # id_single (sorts last -> highest pid) is ineligible for holdout and stays in train.
    single_pid = int(split.pids.max())
    assert single_pid not in set(query.pids.tolist())
    assert single_pid in set(train.pids.tolist())


def test_holdout_preserves_train_camids(tmp_path: Path) -> None:
    _make_crops(tmp_path, num_ids=6, num_per=4)
    train, _, _ = build_crop_identity_holdout_split(load_identity_crops(tmp_path), seed=1)
    # Train crops keep the loader's placeholder camids.
    assert set(train.camids.tolist()) == {-1}


def test_holdout_rejects_bad_fraction(tmp_path: Path) -> None:
    _make_crops(tmp_path, num_ids=4, num_per=4)
    split = load_identity_crops(tmp_path)

    with pytest.raises(ValueError, match="holdout_fraction must be in"):
        build_crop_identity_holdout_split(split, holdout_fraction=0.0)
    with pytest.raises(ValueError, match="holdout_fraction must be in"):
        build_crop_identity_holdout_split(split, holdout_fraction=1.5)
