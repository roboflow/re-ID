# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Identity-folder datasets."""

from __future__ import annotations

from pathlib import Path

import pytest

from reid.data import ReIDDataset, ReIDSplit

from .crop_images import write_identity_crops


class TestReIDDataset:
    """Folder scanning: dense person IDs and empty-input errors."""

    def test_numbers_identities_in_folder_order(self, tmp_path: Path) -> None:
        write_identity_crops(tmp_path, num_ids=3, num_per=4)
        split = ReIDDataset(tmp_path).split

        assert isinstance(split, ReIDSplit)
        assert split.pids.tolist() == [0] * 4 + [1] * 4 + [2] * 4
        # Crops on disk carry no camera information.
        assert set(split.camids.tolist()) == {-1}

    def test_reports_its_size(self, tmp_path: Path) -> None:
        write_identity_crops(tmp_path, num_ids=2, num_per=3)
        dataset = ReIDDataset(tmp_path)

        assert len(dataset) == 6
        assert dataset.num_identities == 2
        assert "images=6" in repr(dataset)
        assert "identities=2" in repr(dataset)

    def test_skips_empty_folders_directories_and_other_files(self, tmp_path: Path) -> None:
        folder = tmp_path / "identity"
        (folder / "directory.jpg").mkdir(parents=True)
        image = folder / "image.PNG"
        image.touch()
        (folder / "notes.txt").touch()
        (tmp_path / "empty").mkdir()

        assert ReIDDataset(tmp_path).split.image_paths == [str(image)]

    def test_missing_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Crop dataset root not found"):
            ReIDDataset(tmp_path / "nope")

    def test_no_crops_raises(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="No identity crops found"):
            ReIDDataset(tmp_path)
