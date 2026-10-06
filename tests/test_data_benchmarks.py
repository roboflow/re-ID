# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Market-1501 and MSMT17 benchmark loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from reid.data import load_market1501, load_msmt17


def test_market1501_reads_person_ids_from_file_names(tmp_path: Path) -> None:
    query_dir = tmp_path / "query"
    gallery_dir = tmp_path / "bounding_box_test"
    query_dir.mkdir()
    gallery_dir.mkdir()
    # The loaders read identities and cameras from file names only, so placeholder bytes are enough.
    (query_dir / "0001_c1s1_001051_00.jpg").write_bytes(b"jpeg")
    (gallery_dir / "0000_c1s1_000151_01.jpg").write_bytes(b"jpeg")
    (gallery_dir / "0002_c2s1_000851_01.jpg").write_bytes(b"jpeg")

    query, gallery = load_market1501(tmp_path)
    assert len(query) == 1
    assert query.pids.tolist() == [1]
    assert gallery.pids.tolist() == [0, 2]
    # Distractors (person 0) stay wrong candidates: torchreid's market1501.py drops only pid -1 images.
    assert gallery.gallery_junk_pids == frozenset({-1})


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

    query, gallery = load_msmt17(tmp_path)

    assert query.pids.tolist() == [42]  # the person ID comes from the list line
    assert query.camids.tolist() == [6]  # camera field 07 counts from 1
    assert gallery.camids.tolist() == [6]
    assert gallery.gallery_junk_pids == frozenset({-1})


def test_msmt17_rejects_list_lines_with_extra_columns(tmp_path: Path) -> None:
    _write_msmt17(tmp_path, list_line=f"{_MSMT17_IMAGE} 42 7")  # a camera column the official lists do not have

    with pytest.raises(ValueError, match="list_query.txt:1: expected 'path pid'"):
        load_msmt17(tmp_path)
