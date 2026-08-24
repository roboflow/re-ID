# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for MOT GT patch generation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from reid.data.mot import generate_mot_patches, sequence_qualified_track_key

_FRAME_H, _FRAME_W = 128, 64


def test_sequence_qualified_track_key_disambiguates_raw_ids() -> None:
    """Raw MOT track IDs restart per sequence; keys must not collide."""
    assert sequence_qualified_track_key("MOT17-02-FRCNN", 1) != sequence_qualified_track_key("MOT17-04-FRCNN", 1)
    assert sequence_qualified_track_key("MOT17-02-FRCNN", 1) == "MOT17-02-FRCNN_1"


def _write_sequence(
    root: Path,
    name: str,
    seq_length: int,
    rows: list[list[float]],
    *,
    frame_width: int = 6,
) -> Path:
    """Create a minimal MOT sequence (seqinfo.ini, img1/ frames, gt/gt.txt)."""
    sequence_dir = root / name
    (sequence_dir / "img1").mkdir(parents=True)
    (sequence_dir / "gt").mkdir(parents=True)

    (sequence_dir / "seqinfo.ini").write_text(f"[Sequence]\nname={name}\nseqLength={seq_length}\n")

    for frame in range(1, seq_length + 1):
        image = np.full((_FRAME_H, _FRAME_W, 3), frame % 256, dtype=np.uint8)
        cv2.imwrite(str(sequence_dir / "img1" / f"{frame:0{frame_width}d}.jpg"), image)

    lines = [",".join(str(value) for value in row) for row in rows]
    (sequence_dir / "gt" / "gt.txt").write_text("\n".join(lines) + "\n")
    return sequence_dir


def _box(frame: int, identity: int, *, conf: int = 1, cls: int = 1, vis: float = 1.0):
    return [frame, identity, 5, 5, 20, 40, conf, cls, vis]


def _pedestrian_rows(seq_length: int, identities: list[int]) -> list[list[float]]:
    return [_box(frame, identity) for identity in identities for frame in range(1, seq_length + 1)]


class TestGenerateMotPatches:
    """Tests for :func:`generate_mot_patches`."""

    def test_train_half_is_frame_disjoint_from_val_half(self, tmp_path: Path) -> None:
        mot_root = tmp_path / "MOT"
        _write_sequence(mot_root, "SEQ-01", seq_length=4, rows=_pedestrian_rows(4, [1, 2]))

        train_out = tmp_path / "train"
        val_out = tmp_path / "val"
        train_stats = generate_mot_patches(mot_root, train_out, split="train_half")
        val_stats = generate_mot_patches(mot_root, val_out, split="val_half")

        def crop_frames(out: Path) -> set[int]:
            return {int(p.stem.split("_")[-1]) for p in out.rglob("*.jpg")}

        assert crop_frames(train_out) == {1, 2}
        assert crop_frames(val_out) == {3, 4}
        assert train_stats.num_crops == 4
        assert val_stats.num_crops == 4
        assert train_stats.num_identities == 2

    def test_identity_folders_are_globally_unique_across_sequences(self, tmp_path: Path) -> None:
        mot_root = tmp_path / "MOT"
        _write_sequence(mot_root, "SEQ-01", seq_length=2, rows=_pedestrian_rows(2, [1]))
        _write_sequence(mot_root, "SEQ-02", seq_length=2, rows=_pedestrian_rows(2, [1]))

        out = tmp_path / "out"
        stats = generate_mot_patches(mot_root, out, split="full")

        identity_dirs = sorted(p.name for p in out.iterdir() if p.is_dir())
        assert identity_dirs == ["SEQ-01_1", "SEQ-02_1"]
        assert stats.num_identities == 2

    def test_filters_distractors_visibility_and_small_crops(self, tmp_path: Path) -> None:
        mot_root = tmp_path / "MOT"
        rows = [
            _box(1, 1),
            _box(2, 1),
            _box(1, 2, cls=2),
            _box(2, 3, conf=0),
            _box(1, 4, vis=0.1),
            [1, 5, 5, 5, 4, 40, 1, 1, 1.0],  # width 4 -> dropped by min_side
        ]
        _write_sequence(mot_root, "SEQ-01", seq_length=2, rows=rows)

        out = tmp_path / "out"
        stats = generate_mot_patches(mot_root, out, split="full", min_visibility=0.5, min_side=10)

        assert stats.num_identities == 1
        assert sorted(p.name for p in out.iterdir() if p.is_dir()) == ["SEQ-01_1"]
        assert stats.num_crops == 2
        assert stats.skipped_low_visibility == 1
        assert stats.skipped_small == 1

    def test_detector_variants_are_deduplicated(self, tmp_path: Path) -> None:
        mot_root = tmp_path / "MOT"
        _write_sequence(mot_root, "MOT17-02-FRCNN", seq_length=2, rows=_pedestrian_rows(2, [1]))
        _write_sequence(mot_root, "MOT17-02-SDP", seq_length=2, rows=_pedestrian_rows(2, [1]))

        out = tmp_path / "out"
        stats = generate_mot_patches(mot_root, out, split="full")

        assert stats.sequences == ["MOT17-02"]
        assert stats.num_identities == 1
        assert stats.num_crops == 2

    def test_dedupe_false_keeps_separate_detector_identity_folders(self, tmp_path: Path) -> None:
        mot_root = tmp_path / "MOT"
        _write_sequence(mot_root, "MOT17-02-FRCNN", seq_length=2, rows=_pedestrian_rows(2, [1]))
        _write_sequence(mot_root, "MOT17-02-SDP", seq_length=2, rows=_pedestrian_rows(2, [1]))

        out = tmp_path / "out"
        stats = generate_mot_patches(mot_root, out, split="full", dedupe_detectors=False)

        identity_dirs = sorted(p.name for p in out.iterdir() if p.is_dir())
        assert identity_dirs == ["MOT17-02-FRCNN_1", "MOT17-02-SDP_1"]
        assert stats.num_identities == 2
        assert stats.num_crops == 4
        assert set(stats.sequences) == {"MOT17-02-FRCNN", "MOT17-02-SDP"}

    def test_reads_eight_digit_frame_stems(self, tmp_path: Path) -> None:
        mot_root = tmp_path / "MOT"
        _write_sequence(
            mot_root,
            "dancetrack0001",
            seq_length=2,
            rows=_pedestrian_rows(2, [1]),
            frame_width=8,
        )

        out = tmp_path / "out"
        stats = generate_mot_patches(mot_root, out, split="full")

        assert stats.num_identities == 1
        assert stats.num_crops == 2
        assert stats.skipped_degenerate == 0

    def test_wrong_image_ext_raises_instead_of_empty_output(self, tmp_path: Path) -> None:
        mot_root = tmp_path / "MOT"
        _write_sequence(mot_root, "SEQ-01", seq_length=2, rows=_pedestrian_rows(2, [1]))

        with pytest.raises(RuntimeError, match="image_ext='png'"):
            generate_mot_patches(mot_root, tmp_path / "out", split="full", image_ext="png")

    def test_half_split_requires_seqinfo(self, tmp_path: Path) -> None:
        mot_root = tmp_path / "MOT"
        sequence_dir = _write_sequence(mot_root, "SEQ-01", seq_length=4, rows=_pedestrian_rows(4, [1, 2]))
        (sequence_dir / "seqinfo.ini").unlink()

        with pytest.raises(FileNotFoundError, match="seqLength"):
            generate_mot_patches(mot_root, tmp_path / "out", split="train_half")

    def test_min_visibility_requires_visibility_column(self, tmp_path: Path) -> None:
        mot_root = tmp_path / "MOT"
        rows = [row[:8] for row in _pedestrian_rows(2, [1])]
        _write_sequence(mot_root, "SEQ-01", seq_length=2, rows=rows)

        with pytest.raises(ValueError, match="visibility column"):
            generate_mot_patches(mot_root, tmp_path / "out", split="full", min_visibility=0.5)
