# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Market-1501 and MSMT17 benchmark loaders."""

from __future__ import annotations

from pathlib import Path

from reid.data import MARKET1501_GALLERY_JUNK_PIDS, load_market1501, load_msmt17


def test_load_market1501_from_temp_tree(tmp_path: Path) -> None:
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
    assert gallery.gallery_junk_pids == MARKET1501_GALLERY_JUNK_PIDS


def test_load_msmt17_from_temp_lists(tmp_path: Path) -> None:
    rel = "0001/0001_019_07_0303morning_0020_1.jpg"
    image_path = tmp_path / "test" / rel
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"jpeg")
    # 2-col: camid parsed from the filename camera field (07 → 6).
    # 3-col: camid column is 1-indexed like the filename (07 → 6), so both
    # formats resolve the same physical camera to the same 0-indexed id.
    (tmp_path / "list_query.txt").write_text(f"{rel} 42\n")
    (tmp_path / "list_gallery.txt").write_text(f"{rel} 42 7\n")

    query, gallery = load_msmt17(tmp_path)
    assert query.pids.tolist() == [42]
    assert query.camids.tolist() == [6]
    assert gallery.pids.tolist() == [42]
    assert gallery.camids.tolist() == [6]
    assert gallery.gallery_junk_pids == frozenset({-1})
