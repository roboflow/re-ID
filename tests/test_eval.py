# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""NumPy-only ReID gallery evaluation tests.

These do not require torch. CI installs via ``uv sync --all-extras`` and runs
``pytest tests/ -m 'not slow'``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import supervision as sv
from PIL import Image

from reid import ReIDDataset, ReIDModel


def _make_identity_crops(root: Path, num_ids: int, num_per: int) -> None:
    """Write a ``<root>/<identity>/<crop>.jpg`` tree of random images."""
    rng = np.random.default_rng(0)
    for i in range(num_ids):
        identity_dir = root / f"id_{i:02d}"
        identity_dir.mkdir(parents=True)
        for j in range(num_per):
            array = (rng.random((48, 24, 3)) * 255).astype("uint8")
            Image.fromarray(array).save(identity_dir / f"{j}.jpg")


class TestModelEvaluate:
    """``ReIDModel.evaluate`` derives a partition or takes an explicit one."""

    def test_evaluate_derives_or_accepts_an_explicit_partition(self, tmp_path: Path) -> None:
        """Defaults handle the split; the derivation knobs are reachable; explicit splits win."""
        data_root = tmp_path / "crops"
        _make_identity_crops(data_root, num_ids=6, num_per=6)
        data = ReIDDataset(data_root)
        model = ReIDModel.from_pretrained(architecture="osnet_x0_25", device="cpu")

        assert model.evaluate(data, verbose=False).metrics.num_queries == 6
        assert model.evaluate(data_root, verbose=False).metrics.num_queries == 6
        assert model.evaluate(data, queries_per_id=2, min_crops=3, verbose=False).metrics.num_queries == 12

        query, gallery = data.for_retrieval(queries_per_id=2, min_crops=3, seed=7)
        assert model.evaluate(query, gallery, verbose=False).metrics.num_queries == 12

    def test_evaluate_rejects_derivation_args_with_an_explicit_gallery(self, tmp_path: Path) -> None:
        """Silently ignoring a split knob the caller set would hide a mistake."""
        data_root = tmp_path / "crops"
        _make_identity_crops(data_root, num_ids=6, num_per=6)
        query, gallery = ReIDDataset(data_root).for_retrieval()
        model = ReIDModel.from_pretrained(architecture="osnet_x0_25", device="cpu")

        with pytest.raises(ValueError, match="only applies when the partition is derived"):
            model.evaluate(query, gallery, seed=7)


class TestDistanceMatrix:
    """Query x gallery distance matrix in cosine and euclidean metrics."""

    def test_cosine_and_euclidean_shapes(self) -> None:
        from reid.eval.evaluator import _distance_matrix

        q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        g = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        cos = _distance_matrix(q, g, "cosine")
        euc = _distance_matrix(q, g, "euclidean")
        assert cos.shape == (2, 2)
        assert euc.shape == (2, 2)
        assert np.allclose(cos.diagonal(), 0.0, atol=1e-5)


class TestComputeReidMetrics:
    """mAP / CMC / mINP computation and input validation."""

    def test_market1501_pid_zero_is_junk_in_gallery(self) -> None:
        from reid.data import MARKET1501_GALLERY_JUNK_PIDS
        from reid.eval.metrics import compute_reid_metrics

        metrics = compute_reid_metrics(
            np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
            q_pids=np.array([1]),
            g_pids=np.array([0, 1, 2]),
            q_camids=np.array([0]),
            g_camids=np.array([0, 1, 0]),
            max_rank=10,
            gallery_junk_pids=MARKET1501_GALLERY_JUNK_PIDS,
        )
        assert metrics.rank1 == pytest.approx(100.0)

    def test_cmc_pads_when_valid_gallery_shorter_than_max_rank(self) -> None:
        from reid.eval.metrics import compute_reid_metrics

        metrics = compute_reid_metrics(
            np.array([[0.2, 0.1]], dtype=np.float32),
            q_pids=np.array([3]),
            g_pids=np.array([3, 8]),
            q_camids=np.array([0]),
            g_camids=np.array([1, 0]),
            max_rank=10,
        )
        assert metrics.rank1 == pytest.approx(0.0)
        assert metrics.rank5 == pytest.approx(100.0)

    def test_max_rank_below_ten_raises(self) -> None:
        from reid.eval.metrics import compute_reid_metrics

        with pytest.raises(ValueError, match="max_rank"):
            compute_reid_metrics(
                np.array([[0.1, 0.2]], dtype=np.float32),
                q_pids=np.array([1]),
                g_pids=np.array([1, 2]),
                q_camids=np.array([0]),
                g_camids=np.array([1, 0]),
                max_rank=5,
            )

    def test_computes_map_cmc_and_minp_on_controlled_ranking(self) -> None:
        """One query, two true gallery matches after junk removal.

        Distance order of gallery ids ``[2, 1, 1, 1]`` with cams ``[1, 1, 0, 2]``.
        Same-(pid, camid) as the query (third gallery entry) is junk, so the
        valid ranking is distractor → match → match.

        Hand-computed: Rank-1 = 0%, AP = 7/12, mINP = 2/3.
        """
        from reid.eval.metrics import compute_reid_metrics

        metrics = compute_reid_metrics(
            np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
            q_pids=np.array([1]),
            g_pids=np.array([2, 1, 1, 1]),
            q_camids=np.array([0]),
            g_camids=np.array([1, 1, 0, 2]),
            max_rank=10,
            gallery_junk_pids=frozenset(),
        )
        assert metrics.num_queries == 1
        assert metrics.rank1 == pytest.approx(0.0)
        assert metrics.rank5 == pytest.approx(100.0)
        assert metrics.rank10 == pytest.approx(100.0)
        assert metrics.mean_average_precision == pytest.approx((7.0 / 12.0) * 100.0)
        assert metrics.minp == pytest.approx((2.0 / 3.0) * 100.0)


class TestReIDEvaluator:
    """``ReIDEvaluator`` end-to-end retrieval scoring."""

    def test_reuses_provided_query_embeddings_only(self, tmp_path: Path) -> None:
        from reid.data import ReIDSplit
        from reid.eval.evaluator import ReIDEvaluator

        query_img = tmp_path / "q.jpg"
        gallery_img = tmp_path / "g.jpg"
        query_img.write_bytes(b"jpeg")
        gallery_img.write_bytes(b"jpeg")
        query = ReIDSplit(
            image_paths=[str(query_img)],
            pids=np.array([1]),
            camids=np.array([0]),
        )
        gallery = ReIDSplit(
            image_paths=[str(gallery_img)],
            pids=np.array([1]),
            camids=np.array([1]),
        )

        class _Encoder:
            def __init__(self) -> None:
                self.calls: list[list[str]] = []

            def extract_features(self, detections: sv.Detections, frame: np.ndarray) -> np.ndarray:
                raise NotImplementedError

            def extract_features_from_paths(
                self,
                image_paths: Sequence[str],
                *,
                batch_size: int = 64,
            ) -> np.ndarray:
                self.calls.append(list(image_paths))
                return np.ones((len(image_paths), 2), dtype=np.float32)

        encoder = _Encoder()
        provided_query = np.array([[0.0, 1.0]], dtype=np.float32)
        result = ReIDEvaluator(encoder).evaluate(
            query,
            gallery,
            query_embeddings=provided_query,
            verbose=False,
        )
        assert encoder.calls == [[str(gallery_img)]]
        np.testing.assert_array_equal(result.query_embeddings, provided_query)

    def test_wrong_length_provided_embeddings_raise(self, tmp_path: Path) -> None:
        from reid.data import ReIDSplit
        from reid.eval.evaluator import ReIDEvaluator

        query_img = tmp_path / "q.jpg"
        gallery_img = tmp_path / "g.jpg"
        query_img.write_bytes(b"jpeg")
        gallery_img.write_bytes(b"jpeg")
        query = ReIDSplit(
            image_paths=[str(query_img)],
            pids=np.array([1]),
            camids=np.array([0]),
        )
        gallery = ReIDSplit(
            image_paths=[str(gallery_img)],
            pids=np.array([1]),
            camids=np.array([1]),
        )

        class _Encoder:
            def extract_features(self, detections: sv.Detections, frame: np.ndarray) -> np.ndarray:
                raise NotImplementedError

            def extract_features_from_paths(
                self,
                image_paths: Sequence[str],
                *,
                batch_size: int = 64,
            ) -> np.ndarray:
                return np.ones((len(image_paths), 2), dtype=np.float32)

        evaluator = ReIDEvaluator(_Encoder())
        with pytest.raises(ValueError, match="query_embeddings"):
            evaluator.evaluate(
                query,
                gallery,
                query_embeddings=np.ones((2, 2), dtype=np.float32),
                verbose=False,
            )
        with pytest.raises(ValueError, match="gallery_embeddings"):
            evaluator.evaluate(
                query,
                gallery,
                gallery_embeddings=np.ones((3, 2), dtype=np.float32),
                verbose=False,
            )


class TestMarket1501Loader:
    """Market-1501 folder parsing."""

    def test_load_market1501_from_temp_tree(self, tmp_path) -> None:
        from reid.data import (
            MARKET1501_GALLERY_JUNK_PIDS,
            load_market1501,
        )

        query_dir = tmp_path / "query"
        gallery_dir = tmp_path / "bounding_box_test"
        query_dir.mkdir()
        gallery_dir.mkdir()
        (query_dir / "0001_c1s1_001051_00.jpg").write_bytes(b"jpeg")
        (gallery_dir / "0000_c1s1_000151_01.jpg").write_bytes(b"jpeg")
        (gallery_dir / "0002_c2s1_000851_01.jpg").write_bytes(b"jpeg")

        query, gallery = load_market1501(tmp_path)
        assert len(query) == 1
        assert query.pids.tolist() == [1]
        assert gallery.pids.tolist() == [0, 2]
        assert gallery.gallery_junk_pids == MARKET1501_GALLERY_JUNK_PIDS


class TestMSMT17Loader:
    """MSMT17 list-file parsing."""

    def test_load_msmt17_from_temp_lists(self, tmp_path) -> None:
        from reid.data import load_msmt17

        root = Path(tmp_path)
        test_root = root / "test"
        test_root.mkdir()
        rel = "0001/0001_019_07_0303morning_0020_1.jpg"
        image_path = test_root / rel
        image_path.parent.mkdir(parents=True)
        image_path.write_bytes(b"jpeg")
        # 2-col: camid parsed from the filename camera field (07 → 6).
        # 3-col: camid column is 1-indexed like the filename (07 → 6), so both
        # formats resolve the same physical camera to the same 0-indexed id.
        (root / "list_query.txt").write_text(f"{rel} 42\n")
        (root / "list_gallery.txt").write_text(f"{rel} 42 7\n")

        query, gallery = load_msmt17(root)
        assert query.pids.tolist() == [42]
        assert query.camids.tolist() == [6]
        assert gallery.pids.tolist() == [42]
        assert gallery.camids.tolist() == [6]
        assert gallery.gallery_junk_pids == frozenset({-1})
