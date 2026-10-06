# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Evaluation protocols: selection, settings, and the output contract every protocol shares."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reid.data import ReIDSplit, load_identity_folders
from reid.eval import EvaluationProtocol, ReassociationProtocol, RetrievalProtocol
from reid.eval.metrics import ReIDMetrics

from .crop_images import write_identity_crops


def _split(paths: list[str], pids: list[int], camids: list[int]) -> ReIDSplit:
    return ReIDSplit(image_paths=list(paths), pids=np.array(pids), camids=np.array(camids))


_ONE_IMAGE = _split(["q"], pids=[1], camids=[0])
_EMPTY = ReIDSplit(image_paths=[], pids=np.array([], dtype=int), camids=np.array([], dtype=int))


class TestRetrievalSelection:
    """Which images retrieval compares: given splits, or query and gallery cut from a dataset."""

    def test_explicit_splits_are_used_as_given(self) -> None:
        query = _split(["q"], pids=[7], camids=[0])
        gallery = _split(["g"], pids=[7], camids=[1])

        selected_query, selected_gallery, mask, _ = RetrievalProtocol().build_query_gallery(query, gallery)

        assert selected_query.image_paths == ("q",)
        assert selected_gallery.image_paths == ("g",)
        assert mask is None  # every query is ranked against the whole gallery

    def test_cuts_each_identity_into_query_and_gallery(self, tmp_path: Path) -> None:
        write_identity_crops(tmp_path, num_ids=8, num_per=4)

        query, gallery, _, _ = RetrievalProtocol().build_query_gallery(load_identity_folders(tmp_path))

        assert len(query) == 8  # one query per identity
        assert len(gallery) == 8 * 3
        assert set(query.pids.tolist()) == set(gallery.pids.tolist())
        # Query and gallery get different synthetic cameras, so no match is dropped as same-camera.
        assert set(query.camids.tolist()) == {0}
        assert set(gallery.camids.tolist()) == {1}

    def test_same_seed_selects_the_same_images(self, tmp_path: Path) -> None:
        """Models and training epochs scored with one protocol see identical inputs."""
        write_identity_crops(tmp_path, num_ids=6, num_per=5)
        crops = load_identity_folders(tmp_path)

        query, _, _, _ = RetrievalProtocol(seed=7).build_query_gallery(crops)
        same_seed, _, _, _ = RetrievalProtocol(seed=7).build_query_gallery(crops)
        other_seed, _, _, _ = RetrievalProtocol(seed=8).build_query_gallery(crops)

        assert same_seed.image_paths == query.image_paths
        assert other_seed.image_paths != query.image_paths

    def test_min_crops_defaults_to_one_more_than_queries_per_id(self, tmp_path: Path) -> None:
        write_identity_crops(tmp_path, num_ids=1, num_per=4)

        query, gallery, _, _ = RetrievalProtocol(queries_per_id=3).build_query_gallery(load_identity_folders(tmp_path))

        assert len(query) == 3  # 4 crops meet the default min_crops of 3 + 1
        assert len(gallery) == 1

    def test_leaves_out_identities_with_too_few_crops(self, tmp_path: Path) -> None:
        write_identity_crops(tmp_path, num_ids=3, num_per=4)
        write_identity_crops(tmp_path, num_ids=1, num_per=1, prefix="single")  # one crop cannot be query and gallery

        query, _, _, _ = RetrievalProtocol().build_query_gallery(load_identity_folders(tmp_path))

        assert len(query) == 3

    def test_needs_an_identity_with_min_crops(self, tmp_path: Path) -> None:
        write_identity_crops(tmp_path, num_ids=3, num_per=2)

        with pytest.raises(ValueError, match="No identity has >= 3 crops"):
            RetrievalProtocol(min_crops=3).build_query_gallery(load_identity_folders(tmp_path))


class TestReportedMetrics:
    """The metrics each protocol reports from the same computed statistics."""

    def test_each_protocol_reports_its_own_metrics(self) -> None:
        metrics = ReIDMetrics(mean_average_precision=40, rank1=50, rank5=60, rank10=70, minp=80, num_queries=1)

        retrieval = RetrievalProtocol().metric_values(metrics)
        reassociation = ReassociationProtocol().metric_values(metrics)

        assert retrieval == {"map": 40, "rank1": 50, "rank5": 60, "rank10": 70, "minp": 80}
        # One correct match per re-association query makes average precision the reciprocal rank.
        assert reassociation == {"rank1": 50, "mrr": 40}


class TestSettings:
    """Invalid protocol settings are rejected before any image is embedded."""

    @pytest.mark.parametrize(
        "protocol_class, setting",
        [
            (RetrievalProtocol, "queries_per_id"),
            (ReassociationProtocol, "gap_frames"),
            (ReassociationProtocol, "query_stride"),
        ],
    )
    def test_count_settings_must_be_positive(self, protocol_class: type[EvaluationProtocol], setting: str) -> None:
        with pytest.raises(ValueError, match=f"{setting} must be a positive integer"):
            protocol_class(**{setting: 0})

    def test_min_crops_must_leave_a_gallery_image(self) -> None:
        with pytest.raises(ValueError, match="min_crops must be > queries_per_id"):
            RetrievalProtocol(queries_per_id=2, min_crops=2)  # both crops would become queries

    def test_unknown_distance_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="distance must be"):
            RetrievalProtocol(distance="manhattan").build_query_gallery(_ONE_IMAGE, _ONE_IMAGE)


class TestInputs:
    """What ``build_query_gallery`` accepts as data and gallery."""

    @pytest.mark.parametrize("protocol_class", [RetrievalProtocol, ReassociationProtocol])
    def test_folder_paths_must_be_loaded_first(self, protocol_class: type[EvaluationProtocol]) -> None:
        with pytest.raises(TypeError, match="needs a ReIDSplit.*got str"):
            protocol_class().build_query_gallery("crops")

    def test_an_explicit_gallery_needs_a_query_split(self) -> None:
        with pytest.raises(TypeError, match="explicit gallery"):
            RetrievalProtocol().build_query_gallery("crops", _ONE_IMAGE)

    def test_sampling_settings_do_not_apply_to_explicit_splits(self) -> None:
        """``seed`` picks query images from a dataset; with given splits it would be silently ignored."""
        with pytest.raises(ValueError, match="only apply when the partition is derived"):
            RetrievalProtocol(seed=1).build_query_gallery(_ONE_IMAGE, _ONE_IMAGE)

    def test_reassociation_builds_its_own_gallery(self) -> None:
        with pytest.raises(ValueError, match="gallery must be omitted"):
            ReassociationProtocol().build_query_gallery(_ONE_IMAGE, _ONE_IMAGE)

    def test_empty_gallery_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="nonempty"):
            RetrievalProtocol().build_query_gallery(_ONE_IMAGE, _EMPTY)
