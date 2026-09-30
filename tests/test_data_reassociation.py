# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Re-association candidate sets built from MOT crop names."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reid.data import ReIDDataset
from reid.eval import ReassociationProtocol
from reid.eval.metrics import _compute_reid_metrics

from .crop_images import write_mot_crop


def _write_crops(root: Path, sequence: str, *, track_ids: tuple[int, ...], frames: tuple[int, ...]) -> None:
    """Write one crop per track at each frame."""
    for track_id in track_ids:
        for frame in frames:
            write_mot_crop(root, sequence, track_id, frame)


def _track_and_frame(path: str) -> tuple[int, int]:
    """Read the (track, frame) of a crop written by ``write_mot_crop``, such as ``S_2/S_000030.jpg``."""
    path = Path(path)
    return int(path.parent.name.rsplit("_", 1)[1]), int(path.stem.rsplit("_", 1)[1])


def test_candidates_are_deterministic_and_stay_in_their_sequence(tmp_path: Path) -> None:
    for sequence in ("seq_B", "seq_A"):  # underscores inside sequence names must survive parsing
        _write_crops(tmp_path, sequence, track_ids=(2, 1), frames=(0, 30))
    query, gallery, mask, _ = ReassociationProtocol().build_query_gallery(ReIDDataset(tmp_path))
    again_query, again_gallery, again_mask, _ = ReassociationProtocol().build_query_gallery(ReIDDataset(tmp_path))
    assert query.image_paths == again_query.image_paths
    assert gallery.image_paths == again_gallery.image_paths
    assert mask.tolist() == again_mask.tolist()
    assert len(query) == 4
    assert len(gallery) == 4
    assert query.pids.tolist() == [0, 1, 2, 3]
    # Each frame-0 query ranks only the two tracks of its own sequence at frame 30.
    assert mask.sum(axis=1).tolist() == [2, 2, 2, 2]
    for query_path, eligible in zip(query.image_paths, mask):
        sequence = Path(query_path).parent.name.rsplit("_", 1)[0]
        assert all(
            Path(path).parent.name.rsplit("_", 1)[0] == sequence
            for path, included in zip(gallery.image_paths, eligible)
            if included
        )


def test_skipped_queries_are_counted_by_reason(tmp_path: Path) -> None:
    _write_crops(tmp_path, "S", track_ids=(1, 2, 3), frames=(0,))
    _write_crops(tmp_path, "S", track_ids=(1, 2), frames=(30,))
    _write_crops(tmp_path, "S", track_ids=(3,), frames=(60,))
    query, _, _, metadata = ReassociationProtocol(query_stride=30).build_query_gallery(ReIDDataset(tmp_path))
    # With a 30-frame gap: at frame 0, tracks 1 and 2 reappear at frame 30 and are scored, while track 3 does not
    # reappear (absent positive). At frame 30, tracks 1 and 2 face a single person at frame 60 (insufficient
    # candidates). At frame 60, track 3 has no frame 90 at all (missing future).
    assert len(query) == 2
    assert metadata["skipped_missing_future"] == 1
    assert metadata["skipped_absent_positive"] == 1
    assert metadata["skipped_insufficient_candidates"] == 2
    assert metadata["num_skipped_queries"] == 4


def test_each_query_is_compared_with_the_people_exactly_gap_frames_later(tmp_path: Path) -> None:
    """Queries at frames 0 and 60 see only the crops 30 frames later, at 30 and 90.

    Crops at frames 29 and 31 sit one frame either side of that gap and must never be candidates.
    """
    _write_crops(tmp_path, "S", track_ids=(1, 2), frames=(0, 29, 30, 31, 60, 90))

    query, gallery, mask, _ = ReassociationProtocol(gap_frames=30, query_stride=60).build_query_gallery(
        ReIDDataset(tmp_path)
    )

    # (track, frame) of each query, mapped to the (track, frame) of the crops it is compared with.
    candidates = {
        _track_and_frame(query_path): sorted(
            _track_and_frame(path) for path, keep in zip(gallery.image_paths, row) if keep
        )
        for query_path, row in zip(query.image_paths, mask)
    }
    assert candidates == {
        (1, 0): [(1, 30), (2, 30)],
        (2, 0): [(1, 30), (2, 30)],
        (1, 60): [(1, 90), (2, 90)],
        (2, 60): [(1, 90), (2, 90)],
    }


def test_mrr_gives_partial_credit_to_matches_below_rank_1(tmp_path: Path) -> None:
    """Two queries: one finds its match first, the other second, so rank-1 is 50% and MRR is (1 + 1/2) / 2 = 75%."""
    _write_crops(tmp_path, "S", track_ids=(1, 2), frames=(0, 30))
    query, gallery, mask, _ = ReassociationProtocol().build_query_gallery(ReIDDataset(tmp_path))

    assert [_track_and_frame(path) for path in query.image_paths] == [(1, 0), (2, 0)]
    assert [_track_and_frame(path) for path in gallery.image_paths] == [(1, 30), (2, 30)]
    # Rows are the two queries, columns the two candidates, both in track order.
    distances = np.array(
        [
            [0.1, 0.5],  # track 1's query: its own crop is nearest, so the match is at rank 1
            [0.2, 0.4],  # track 2's query: track 1's crop is nearer, so the match is at rank 2
        ],
        dtype=np.float32,
    )
    metrics = _compute_reid_metrics(
        distances,
        q_pids=query.pids,
        g_pids=gallery.pids,
        q_camids=query.camids,
        g_camids=gallery.camids,
        candidate_mask=mask,
        exclude_same_pid_camid=ReassociationProtocol.exclude_same_pid_camid,
    )

    assert metrics.rank1 == pytest.approx(50.0)  # one of the two queries is right at rank 1
    assert ReassociationProtocol().metric_values(metrics)["mrr"] == pytest.approx(75.0)


def test_no_scorable_query_raises(tmp_path: Path) -> None:
    _write_crops(tmp_path, "S", track_ids=(1, 2), frames=(0,))
    with pytest.raises(ValueError, match="No re-association queries"):
        ReassociationProtocol().build_query_gallery(ReIDDataset(tmp_path))


def test_duplicate_frame_record_rejected(tmp_path: Path) -> None:
    _write_crops(tmp_path, "S", track_ids=(1, 2), frames=(0, 30))
    (tmp_path / "S_1" / "S_000000.png").touch()  # a second crop of track 1 at frame 0
    with pytest.raises(ValueError, match="Duplicate"):
        ReassociationProtocol().build_query_gallery(ReIDDataset(tmp_path))


@pytest.mark.parametrize(
    "directory, filename, match",
    [
        ("broken", "broken_0.jpg", "Malformed MOT crop identity folder"),
        ("S_1", "T_0.jpg", "sequence mismatch"),
        ("S_1", "S_bad.jpg", "sequence mismatch"),
    ],
    ids=["folder_without_track_id", "other_sequence", "frame_not_a_number"],
)
def test_malformed_names_rejected(tmp_path: Path, directory: str, filename: str, match: str) -> None:
    (tmp_path / directory).mkdir()
    (tmp_path / directory / filename).touch()
    with pytest.raises(ValueError, match=match):
        ReassociationProtocol().build_query_gallery(ReIDDataset(tmp_path))
