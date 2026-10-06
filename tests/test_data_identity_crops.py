# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Identity-folder datasets."""

from __future__ import annotations

from pathlib import Path

import pytest

from reid.data import ReIDSplit, load_identity_folders

from .crop_images import write_identity_crops


class TestLoadIdentityFolders:
    """Folder scanning: dense person IDs and empty-input errors."""

    def test_numbers_identities_in_folder_order(self, tmp_path: Path) -> None:
        write_identity_crops(tmp_path, num_ids=3, num_per=4)
        split = load_identity_folders(tmp_path)

        assert isinstance(split, ReIDSplit)
        assert split.pids.tolist() == [0] * 4 + [1] * 4 + [2] * 4
        # Crops on disk carry no camera information.
        assert set(split.camids.tolist()) == {-1}

    def test_skips_empty_folders_directories_and_other_files(self, tmp_path: Path) -> None:
        folder = tmp_path / "identity"
        (folder / "directory.jpg").mkdir(parents=True)
        image = folder / "image.PNG"
        image.touch()
        (folder / "notes.txt").touch()
        (tmp_path / "empty").mkdir()

        assert load_identity_folders(tmp_path).image_paths == [str(image)]

    def test_missing_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Crop dataset root not found"):
            load_identity_folders(tmp_path / "nope")

    def test_no_crops_raises(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="No identity crops found"):
            load_identity_folders(tmp_path)
