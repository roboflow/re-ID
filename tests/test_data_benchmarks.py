# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Market-1501 and MSMT17 benchmark loaders."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from reid.data import (
    ReIDSplit,
    load_market1501_test,
    load_market1501_train,
    load_msmt17_test,
    load_msmt17_train,
    load_msmt17_val,
)


def test_market1501_reads_person_ids_from_file_names(tmp_path: Path) -> None:
    query_dir = tmp_path / "query"
    gallery_dir = tmp_path / "bounding_box_test"
    query_dir.mkdir()
    gallery_dir.mkdir()
    # The loaders read identities and cameras from file names only, so placeholder bytes are enough.
    (query_dir / "0001_c1s1_001051_00.jpg").write_bytes(b"jpeg")
    (gallery_dir / "0000_c1s1_000151_01.jpg").write_bytes(b"jpeg")
    (gallery_dir / "0002_c2s1_000851_01.jpg").write_bytes(b"jpeg")

    query, gallery = load_market1501_test(tmp_path)
    assert len(query) == 1
    assert query.pids.tolist() == [1]
    assert gallery.pids.tolist() == [0, 2]
    # Distractors (person 0) stay wrong candidates: torchreid's market1501.py drops only pid -1 images.
    assert gallery.gallery_junk_pids == frozenset({-1})


def test_market1501_train_reads_the_training_folder(tmp_path: Path) -> None:
    train_dir = tmp_path / "bounding_box_train"  # Market-1501's name for its training crops
    train_dir.mkdir()
    (train_dir / "0002_c1s1_000451_03.jpg").write_bytes(b"jpeg")
    (train_dir / "0007_c3s3_077419_03.jpg").write_bytes(b"jpeg")

    train = load_market1501_train(tmp_path)

    assert train.pids.tolist() == [2, 7]
    assert train.camids.tolist() == [0, 2]  # cameras c1 and c3 count from 1


_MSMT17_IMAGE = "0001/0001_019_07_0303morning_0020_1.jpg"  # the third field, 07, is the camera


def _write_msmt17(root: Path, list_line: str) -> None:
    """Write one MSMT17 test image, and query and gallery lists that both hold ``list_line``."""
    image_path = root / "test" / _MSMT17_IMAGE
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"jpeg")
    (root / "list_query.txt").write_text(list_line + "\n")
    (root / "list_gallery.txt").write_text(list_line + "\n")


def test_msmt17_reads_the_camera_from_the_file_name(tmp_path: Path) -> None:
    _write_msmt17(tmp_path, list_line=f"{_MSMT17_IMAGE} 42")

    query, gallery = load_msmt17_test(tmp_path)

    assert query.pids.tolist() == [42]  # the person ID comes from the list line
    assert query.camids.tolist() == [6]  # camera field 07 counts from 1
    assert gallery.camids.tolist() == [6]
    assert gallery.gallery_junk_pids == frozenset({-1})


def test_msmt17_rejects_list_lines_with_extra_columns(tmp_path: Path) -> None:
    _write_msmt17(tmp_path, list_line=f"{_MSMT17_IMAGE} 42 7")  # a camera column the official lists do not have

    with pytest.raises(ValueError, match="list_query.txt:1: expected 'path pid'"):
        load_msmt17_test(tmp_path)


_MSMT17_TRAIN_IMAGE = "0000/0000_000_01_0303morning_0015_0.jpg"
_MSMT17_VAL_IMAGE = "0000/0000_007_14_0303morning_0036_0.jpg"


@pytest.mark.parametrize(
    "loader, image", [(load_msmt17_train, _MSMT17_TRAIN_IMAGE), (load_msmt17_val, _MSMT17_VAL_IMAGE)]
)
def test_msmt17_train_and_val_each_read_their_own_list(
    tmp_path: Path, loader: Callable[[Path], ReIDSplit], image: str
) -> None:
    """Both lists name images under ``train/``, and the validation images stay out of the training split.

    torchreid's and fast-reid's ``msmt17.py`` read ``list_val.txt`` from ``train_dir`` and add it to the training
    images only under ``combineall``.
    """
    (tmp_path / "train").mkdir()  # the loader reads the lists and never opens an image
    (tmp_path / "list_train.txt").write_text(f"{_MSMT17_TRAIN_IMAGE} 0\n")
    (tmp_path / "list_val.txt").write_text(f"{_MSMT17_VAL_IMAGE} 0\n")

    loaded = loader(tmp_path)

    assert loaded.image_paths == (str(tmp_path / "train" / image),)
