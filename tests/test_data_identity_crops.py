# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for identity-crop loading and split partitioning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from reid.data import (
    ReIDSplit,
    build_identity_crop_holdout_split,
    build_identity_crop_retrieval_split,
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


class TestLoadIdentityCrops:
    """Tests for :func:`load_identity_crops`."""

    def test_assigns_contiguous_pids_and_neg_camids(self, tmp_path: Path) -> None:
        _make_crops(tmp_path, num_ids=3, num_per=4)
        split = load_identity_crops(tmp_path)

        assert isinstance(split, ReIDSplit)
        assert len(split) == 12
        assert sorted(set(split.pids.tolist())) == [0, 1, 2]
        assert set(split.camids.tolist()) == {-1}
        assert split.pids.tolist() == sorted(split.pids.tolist())

    def test_honours_identity_whitelist(self, tmp_path: Path) -> None:
        _make_crops(tmp_path, num_ids=4, num_per=3)
        split = load_identity_crops(tmp_path, identities=["id_00", "id_02"])

        assert sorted(set(split.pids.tolist())) == [0, 1]
        assert len(split) == 6
        assert all(("id_00" in p or "id_02" in p) for p in split.image_paths)


class TestRetrievalSplit:
    """Tests for :func:`build_identity_crop_retrieval_split`."""

    def test_uses_all_eligible_identities(self, tmp_path: Path) -> None:
        _make_crops(tmp_path, num_ids=8, num_per=4)
        query, gallery = build_identity_crop_retrieval_split(load_identity_crops(tmp_path), queries_per_id=1)

        assert len(query) == 8
        assert len(gallery) == 8 * 3
        assert set(query.camids.tolist()) == {0}
        assert set(gallery.camids.tolist()) == {1}
        assert set(query.pids.tolist()) == set(gallery.pids.tolist())

    def test_filters_ineligible_identities(self, tmp_path: Path) -> None:
        _make_crops(tmp_path, num_ids=3, num_per=4)
        (tmp_path / "id_single").mkdir()
        Image.fromarray((np.random.default_rng(1).random((48, 24, 3)) * 255).astype("uint8")).save(
            tmp_path / "id_single" / "00.jpg"
        )
        split = load_identity_crops(tmp_path)

        query, _gallery = build_identity_crop_retrieval_split(split, queries_per_id=1, min_crops=2)
        assert len(query) == 3


class TestHoldoutSplit:
    """Tests for :func:`build_identity_crop_holdout_split`."""

    def test_is_identity_disjoint_from_train(self, tmp_path: Path) -> None:
        _make_crops(tmp_path, num_ids=10, num_per=4)
        train, query, gallery = build_identity_crop_holdout_split(
            load_identity_crops(tmp_path), holdout_fraction=0.2, seed=0
        )

        holdout_pids = set(query.pids.tolist())
        assert set(train.pids.tolist()).isdisjoint(holdout_pids)
        assert len(holdout_pids) == 2
        assert set(query.camids.tolist()) == {0}
        assert set(gallery.camids.tolist()) == {1}
        assert len(query) == len(holdout_pids)
        assert set(query.pids.tolist()) == set(gallery.pids.tolist())

    def test_keeps_ineligible_identities_in_train(self, tmp_path: Path) -> None:
        _make_crops(tmp_path, num_ids=4, num_per=4)
        (tmp_path / "id_single").mkdir()
        Image.fromarray((np.random.default_rng(2).random((48, 24, 3)) * 255).astype("uint8")).save(
            tmp_path / "id_single" / "00.jpg"
        )
        split = load_identity_crops(tmp_path)

        train, query, _ = build_identity_crop_holdout_split(split, holdout_fraction=1.0, min_crops=2, seed=0)
        single_pid = int(split.pids.max())
        assert single_pid not in set(query.pids.tolist())
        assert single_pid in set(train.pids.tolist())
