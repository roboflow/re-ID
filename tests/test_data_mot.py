# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""``generate_mot_patches``: input folder, frame splits, box selection, detector variants, crops, output folder."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from reid.data.mot import generate_mot_patches

_FRAME_HEIGHT, _FRAME_WIDTH = 128, 64


@pytest.fixture
def mot_root(tmp_path: Path) -> Path:
    """Folder the test writes its MOT sequences into."""
    return tmp_path / "MOT"


@pytest.fixture
def output(tmp_path: Path) -> Path:
    """Folder ``generate_mot_patches`` writes crops into. It does not exist until the first crop is written."""
    return tmp_path / "out"


def _box(
    *,
    frame: int,
    track_id: int,
    left: float = 5,
    top: float = 5,
    width: float = 20,
    height: float = 40,
    conf: int = 1,
    cls: int = 1,
    vis: float = 1.0,
) -> list[float]:
    """One MOT ground-truth row: frame, id, left, top, width, height, conf, class, visibility."""
    return [frame, track_id, left, top, width, height, conf, cls, vis]


def _write_sequence(root: Path, name: str, rows: list[list[float]], *, seq_length: int, frame_digits: int = 6) -> Path:
    """Write a MOT sequence: ``seqinfo.ini``, one frame per step in ``img1/``, and ``gt/gt.txt``.

    Frame ``n`` is filled with pixel value ``n``, so a crop shows which frame it came from.
    """
    sequence_dir = root / name
    (sequence_dir / "img1").mkdir(parents=True)
    (sequence_dir / "gt").mkdir(parents=True)
    (sequence_dir / "seqinfo.ini").write_text(f"[Sequence]\nname={name}\nseqLength={seq_length}\n")
    for frame in range(1, seq_length + 1):
        image = np.full((_FRAME_HEIGHT, _FRAME_WIDTH, 3), frame % 256, dtype=np.uint8)
        cv2.imwrite(str(sequence_dir / "img1" / f"{frame:0{frame_digits}d}.jpg"), image)
    (sequence_dir / "gt" / "gt.txt").write_text("\n".join(",".join(map(str, row)) for row in rows) + "\n")
    return sequence_dir


def _write_pedestrians(root: Path, name: str, *, track_ids: list[int], seq_length: int) -> Path:
    """Write a sequence where every track appears in every frame."""
    rows = [_box(frame=frame, track_id=track_id) for track_id in track_ids for frame in range(1, seq_length + 1)]
    return _write_sequence(root, name, rows, seq_length=seq_length)


def _identity_folders(output: Path) -> list[str]:
    """Names of the identity folders written to ``output``, such as ``SEQ-01_1``, sorted."""
    return sorted(path.name for path in output.iterdir())


def _crop_frames(output: Path) -> set[int]:
    """Frames of every crop written to ``output``, read from names such as ``SEQ-01_000003.jpg``."""
    return {int(path.stem.rsplit("_", 1)[1]) for path in output.rglob("*.jpg")}


class TestInputFolder:
    """Finding the sequences under ``mot_root``."""

    def test_missing_root_is_rejected(self, mot_root: Path, output: Path) -> None:
        with pytest.raises(FileNotFoundError, match="MOT root not found"):
            generate_mot_patches(mot_root, output, split="full")
        assert not output.exists()

    def test_sequences_without_ground_truth_are_not_found(self, mot_root: Path, output: Path) -> None:
        sequence_dir = _write_pedestrians(mot_root, "SEQ-01", track_ids=[1], seq_length=2)
        (sequence_dir / "gt" / "gt.txt").unlink()  # like the MOT17 test split: frames, no ground truth

        with pytest.raises(FileNotFoundError, match="No valid MOT sequences"):
            generate_mot_patches(mot_root, output, split="full")
        assert not output.exists()

    def test_requested_sequences_must_exist(self, mot_root: Path, output: Path) -> None:
        _write_pedestrians(mot_root, "SEQ-01", track_ids=[1], seq_length=2)

        with pytest.raises(FileNotFoundError, match=r"not found under .*: \['SEQ-02'\]"):
            generate_mot_patches(mot_root, output, split="full", sequences=["SEQ-01", "SEQ-02"])
        assert not output.exists()


class TestFrameSplits:
    """Which frames of each sequence are cropped."""

    def test_train_and_val_halves_share_no_frame(self, mot_root: Path, tmp_path: Path) -> None:
        # train_half is frames 1 to seqLength // 2, val_half the rest.
        _write_pedestrians(mot_root, "SEQ-01", track_ids=[1, 2], seq_length=4)

        generate_mot_patches(mot_root, tmp_path / "train", split="train_half")
        generate_mot_patches(mot_root, tmp_path / "val", split="val_half")

        assert _crop_frames(tmp_path / "train") == {1, 2}
        assert _crop_frames(tmp_path / "val") == {3, 4}

    def test_half_splits_need_the_sequence_length(self, mot_root: Path, output: Path) -> None:
        sequence_dir = _write_pedestrians(mot_root, "SEQ-01", track_ids=[1], seq_length=2)
        (sequence_dir / "seqinfo.ini").unlink()

        with pytest.raises(FileNotFoundError, match="seqLength"):
            generate_mot_patches(mot_root, output, split="train_half")
        assert not output.exists()

    def test_unknown_split_is_rejected(self, mot_root: Path, output: Path) -> None:
        _write_pedestrians(mot_root, "SEQ-01", track_ids=[1], seq_length=2)

        with pytest.raises(ValueError, match="Unknown split 'test'"):
            generate_mot_patches(mot_root, output, split="test")
        assert not output.exists()


class TestBoxSelection:
    """Which ground-truth boxes become crops, and the ground-truth columns that choice needs."""

    def test_every_class_is_kept_by_default(self, mot_root: Path, output: Path) -> None:
        rows = [_box(frame=1, track_id=1, cls=1), _box(frame=1, track_id=2, cls=2), _box(frame=1, track_id=3, cls=7)]
        _write_sequence(mot_root, "SEQ-01", rows, seq_length=1)

        generate_mot_patches(mot_root, output, split="full")

        assert _identity_folders(output) == ["SEQ-01_1", "SEQ-01_2", "SEQ-01_3"]

    def test_class_ids_keeps_only_the_listed_classes(self, mot_root: Path, output: Path) -> None:
        rows = [_box(frame=1, track_id=1, cls=1), _box(frame=1, track_id=2, cls=2), _box(frame=1, track_id=3, cls=7)]
        _write_sequence(mot_root, "SEQ-01", rows, seq_length=1)

        generate_mot_patches(mot_root, output, split="full", class_ids=[1, 7])

        assert _identity_folders(output) == ["SEQ-01_1", "SEQ-01_3"]  # identity 2 is class 2

    def test_boxes_marked_ignored_are_dropped(self, mot_root: Path, output: Path) -> None:
        rows = [_box(frame=1, track_id=1), _box(frame=1, track_id=2, conf=0)]  # conf 0 marks a box as ignored
        _write_sequence(mot_root, "SEQ-01", rows, seq_length=1)

        generate_mot_patches(mot_root, output, split="full")

        assert _identity_folders(output) == ["SEQ-01_1"]

    def test_min_visibility_drops_occluded_boxes(self, mot_root: Path, output: Path) -> None:
        rows = [_box(frame=1, track_id=1, vis=0.9), _box(frame=1, track_id=2, vis=0.1)]
        _write_sequence(mot_root, "SEQ-01", rows, seq_length=1)

        stats = generate_mot_patches(mot_root, output, split="full", min_visibility=0.5)

        assert _identity_folders(output) == ["SEQ-01_1"]
        assert stats.skipped_low_visibility == 1

    def test_min_side_drops_small_boxes(self, mot_root: Path, output: Path) -> None:
        rows = [_box(frame=1, track_id=1, width=20), _box(frame=1, track_id=2, width=4)]
        _write_sequence(mot_root, "SEQ-01", rows, seq_length=1)

        stats = generate_mot_patches(mot_root, output, split="full", min_side=10)

        assert _identity_folders(output) == ["SEQ-01_1"]
        assert stats.skipped_small == 1

    def test_min_visibility_needs_a_visibility_column(self, mot_root: Path, output: Path) -> None:
        rows = [_box(frame=frame, track_id=1)[:8] for frame in (1, 2)]  # without the visibility column
        _write_sequence(mot_root, "SEQ-01", rows, seq_length=2)

        with pytest.raises(ValueError, match="visibility column"):
            generate_mot_patches(mot_root, output, split="full", min_visibility=0.5)
        assert not output.exists()

    def test_ground_truth_needs_a_class_column(self, mot_root: Path, output: Path) -> None:
        rows = [_box(frame=frame, track_id=1)[:7] for frame in (1, 2)]  # without class and visibility columns
        _write_sequence(mot_root, "SEQ-01", rows, seq_length=2)

        with pytest.raises(ValueError, match="expected >= 8 columns, got 7"):
            generate_mot_patches(mot_root, output, split="full")
        assert not output.exists()


class TestDetectorVariants:
    """MOT17 ships every sequence once per public detector, with the same frames and ground truth."""

    def test_crops_one_copy_of_each_sequence(self, mot_root: Path, output: Path) -> None:
        for detector in ("DPM", "FRCNN", "SDP"):
            _write_pedestrians(mot_root, f"MOT17-02-{detector}", track_ids=[1], seq_length=2)

        stats = generate_mot_patches(mot_root, output, split="full")

        assert stats.sequences == ["MOT17-02"]
        assert _identity_folders(output) == ["MOT17-02_1"]
        assert stats.num_crops == 2  # one per frame, not one per copy


class TestCrops:
    """Crop pixels, frame files, and write failures."""

    def test_crop_holds_the_box_from_its_own_frame(self, mot_root: Path, output: Path) -> None:
        rows = [_box(frame=3, track_id=1, left=3, top=7, width=10, height=30)]
        _write_sequence(mot_root, "SEQ-01", rows, seq_length=3)

        generate_mot_patches(mot_root, output, split="full")

        crop = cv2.imread(str(output / "SEQ-01_1" / "SEQ-01_000003.jpg"))
        assert crop.shape[:2] == (30, 10)  # the box's height and width
        # Frame 3 is filled with pixel value 3; JPEG compression may shift it slightly.
        assert np.abs(crop.astype(int) - 3).max() <= 2

    def test_boxes_crossing_the_frame_edge_are_clipped(self, mot_root: Path, output: Path) -> None:
        # Frames are 64 pixels wide and 128 tall.
        rows = [
            _box(frame=1, track_id=1, left=-5, top=-3, width=20, height=40),  # cut at 0: 15 wide, 37 tall
            _box(frame=1, track_id=2, left=50, top=100, width=30, height=50),  # cut at 64 x 128: 14 wide, 28 tall
        ]
        _write_sequence(mot_root, "SEQ-01", rows, seq_length=1)

        generate_mot_patches(mot_root, output, split="full")

        assert cv2.imread(str(output / "SEQ-01_1" / "SEQ-01_000001.jpg")).shape[:2] == (37, 15)
        assert cv2.imread(str(output / "SEQ-01_2" / "SEQ-01_000001.jpg")).shape[:2] == (28, 14)

    def test_boxes_outside_the_frame_are_skipped(self, mot_root: Path, output: Path) -> None:
        rows = [_box(frame=1, track_id=1), _box(frame=1, track_id=2, left=100)]  # frames are 64 pixels wide
        _write_sequence(mot_root, "SEQ-01", rows, seq_length=1)

        stats = generate_mot_patches(mot_root, output, split="full")

        assert _identity_folders(output) == ["SEQ-01_1"]
        assert stats.skipped_degenerate == 1

    def test_reads_frames_named_with_eight_digits(self, mot_root: Path, output: Path) -> None:
        rows = [_box(frame=frame, track_id=1) for frame in (1, 2)]
        _write_sequence(
            mot_root, "dancetrack0001", rows, seq_length=2, frame_digits=8
        )  # 00000001.jpg, as in DanceTrack

        stats = generate_mot_patches(mot_root, output, split="full")

        assert stats.num_crops == 2

    def test_wrong_image_extension_is_reported(self, mot_root: Path, output: Path) -> None:
        _write_pedestrians(mot_root, "SEQ-01", track_ids=[1], seq_length=2)  # writes .jpg frames

        with pytest.raises(RuntimeError, match="image_ext='png'"):
            generate_mot_patches(mot_root, output, split="full", image_ext="png")  # looks for .png frames
        assert not output.exists()

    def test_failed_write_raises(self, mot_root: Path, output: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_pedestrians(mot_root, "SEQ-01", track_ids=[1], seq_length=2)
        monkeypatch.setattr(cv2, "imwrite", lambda *args: False)

        with pytest.raises(OSError, match="Failed to write MOT crop"):
            generate_mot_patches(mot_root, output, split="full")


class TestOutputFolder:
    """Identity folder names, and which sequences an existing output folder accepts."""

    def test_same_track_id_in_two_sequences_gets_two_folders(self, mot_root: Path, output: Path) -> None:
        # MOT track IDs restart in every sequence, so both sequences have a track 1.
        _write_pedestrians(mot_root, "SEQ-01", track_ids=[1], seq_length=2)
        _write_pedestrians(mot_root, "SEQ-02", track_ids=[1], seq_length=2)

        stats = generate_mot_patches(mot_root, output, split="full")

        assert _identity_folders(output) == ["SEQ-01_1", "SEQ-02_1"]
        assert stats.num_identities == 2  # not 1: the two track 1s are different people

    def test_refuses_crops_of_a_sequence_the_folder_already_holds(self, mot_root: Path, output: Path) -> None:
        _write_pedestrians(mot_root, "SEQ-01", track_ids=[1], seq_length=4)
        generate_mot_patches(mot_root, output, split="val_half")  # crops frames 3 and 4

        # Adding frames 1 and 2 would mix both halves of SEQ-01 in one folder.
        with pytest.raises(FileExistsError, match="already holds crops from .*SEQ-01"):
            generate_mot_patches(mot_root, output, split="train_half")
        assert _crop_frames(output) == {3, 4}

    def test_accepts_crops_of_a_sequence_the_folder_does_not_hold(self, mot_root: Path, output: Path) -> None:
        _write_pedestrians(mot_root, "SEQ-01", track_ids=[1], seq_length=2)
        _write_pedestrians(mot_root, "SEQ-02", track_ids=[1], seq_length=2)
        generate_mot_patches(mot_root, output, split="full", sequences=["SEQ-01"])

        generate_mot_patches(mot_root, output, split="full", sequences=["SEQ-02"])

        assert _identity_folders(output) == ["SEQ-01_1", "SEQ-02_1"]
