# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Distance matrices and retrieval scoring: CMC, mAP, mINP, and candidate masks."""

from __future__ import annotations

import numpy as np
import pytest

from reid.eval.metrics import ReIDMetrics, _compute_reid_metrics, _distance_matrix


@pytest.mark.parametrize("distance, expected", [("cosine", [[0.0, 2.0]]), ("euclidean", [[0.0, 10.0]])])
def test_distance_matrix_values(distance: str, expected: list[list[float]]) -> None:
    # The gallery holds the query and its opposite: cosine distances 0 and 2, Euclidean 0 and 2 * |(3, 4)| = 10.
    query = np.array([[3.0, 4.0]], dtype=np.float32)
    gallery = np.array([[3.0, 4.0], [-3.0, -4.0]], dtype=np.float32)
    assert _distance_matrix(query, gallery, distance) == pytest.approx(np.array(expected), abs=1e-5)


def _score_example(mask: np.ndarray | None = None) -> ReIDMetrics:
    """Score one query, person 1 seen by camera 0, against four gallery images.

    The gallery is listed in order of distance to the query:

    ======  ======  ======  ==========================================
    order   person  camera  counts as
    ======  ======  ======  ==========================================
    1st     1       0       junk: same person and camera, removed
    2nd     2       1       rank 1, wrong person
    3rd     1       1       rank 2, match
    4th     1       2       rank 3, match
    ======  ======  ======  ==========================================
    """
    return _compute_reid_metrics(
        np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        q_pids=np.array([1]),
        g_pids=np.array([1, 2, 1, 1]),
        q_camids=np.array([0]),
        g_camids=np.array([0, 1, 1, 2]),
        candidate_mask=mask,
    )


class TestComputeReidMetrics:
    """Retrieval scores on hand-computed rankings."""

    def test_short_gallery_still_reports_rank5_and_rank10(self) -> None:
        """With two gallery images and the match second, rank 1 misses while rank 5 and rank 10 hit."""
        metrics = _compute_reid_metrics(
            np.array([[0.2, 0.1]], dtype=np.float32),
            q_pids=np.array([3]),
            g_pids=np.array([3, 8]),
            q_camids=np.array([0]),
            g_camids=np.array([1, 0]),
        )
        assert metrics.rank1 == pytest.approx(0.0)
        assert metrics.rank5 == pytest.approx(100.0)
        assert metrics.rank10 == pytest.approx(100.0)

    def test_scores_a_hand_computed_ranking(self) -> None:
        """The ranking in ``_score_example``: wrong person, match, match. Scores are percentages."""
        metrics = _score_example()

        assert metrics.num_queries == 1
        assert metrics.rank1 == pytest.approx(0.0)  # rank 1 is the wrong person
        assert metrics.rank5 == pytest.approx(100.0)
        assert metrics.rank10 == pytest.approx(100.0)
        # AP averages the precision at each match: (1/2 at rank 2 + 2/3 at rank 3) / 2 matches.
        assert metrics.mean_average_precision == pytest.approx((1 / 2 + 2 / 3) / 2 * 100)
        # INP divides the number of matches by the rank of the last one: 2 / 3.
        assert metrics.minp == pytest.approx(2 / 3 * 100)

    def test_junk_filter_is_independent_of_camera_policy(self) -> None:
        """Junk person IDs stay filtered even when the same-camera rule is off."""
        metrics = _compute_reid_metrics(
            np.array([[0.1, 0.2]], dtype=np.float32),
            q_pids=np.array([1]),
            g_pids=np.array([-1, 1]),
            q_camids=np.array([0]),
            g_camids=np.array([0, 0]),
            exclude_same_pid_camid=False,
        )
        assert metrics.rank1 == 100.0


class TestCandidateMask:
    """A candidate mask removes gallery images from both the ranking and the count of correct matches."""

    def test_all_true_mask_matches_no_mask(self) -> None:
        assert _score_example(np.ones((1, 4), dtype=bool)) == _score_example()

    def test_excluded_positive_changes_ap_denominator_and_rank(self) -> None:
        """Masking the rank-2 match (3rd in ``_score_example``) leaves one match, so AP is its 1/2 precision alone.

        The excluded crop is removed from the ranking rather than pushed down it,
        which is what lets a one-positive protocol read average precision as
        reciprocal rank.
        """
        metrics = _score_example(np.array([[True, True, False, True]]))
        assert metrics.num_queries == 1
        assert metrics.mean_average_precision == pytest.approx(50.0)
        assert metrics.rank1 == 0.0
        assert metrics.minp == pytest.approx(50.0)

    def test_all_false_mask_skips_query(self) -> None:
        metrics = _score_example(np.zeros((1, 4), dtype=bool))
        assert metrics.num_queries == 0
        assert metrics.mean_average_precision == 0.0
