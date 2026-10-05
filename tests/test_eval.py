# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""``ReIDModel.evaluate``: protocols, embedding once per image, and input guards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from reid import ReIDDataset, ReIDModel
from reid.data import ReIDSplit
from reid.eval import ReassociationProtocol, RetrievalProtocol
from reid.preprocessing import ReIDPreprocessing

from .crop_images import write_identity_crops, write_mot_crop


def _model() -> ReIDModel:
    """A real ``ReIDModel`` whose embeddings each test supplies or monkeypatches."""
    return ReIDModel(backbone=torch.nn.Identity(), device=torch.device("cpu"), preprocessing=ReIDPreprocessing())


def _model_with(vectors: dict[str, list[float]]) -> ReIDModel:
    """A real ``ReIDModel`` that embeds each image path to a fixed vector."""
    model = _model()
    model.extract_embeddings_from_paths = lambda paths, **kwargs: np.array([vectors[path] for path in paths])
    return model


def test_real_model_evaluates_a_dataset_and_its_explicit_splits(tmp_path: Path) -> None:
    """The default protocol, custom sampling, and explicit splits all run through a real encoder."""
    write_identity_crops(tmp_path, num_ids=6, num_per=6)
    data = ReIDDataset(tmp_path)
    model = ReIDModel.from_pretrained(architecture="osnet_x0_25", device="cpu")

    # One query per identity by default, two when asked: 6 identities give 6 and 12 queries.
    assert model.evaluate(data, verbose=False).num_queries == 6
    assert model.evaluate(data, protocol=RetrievalProtocol(queries_per_id=2), verbose=False).num_queries == 12

    query, gallery, _, _ = RetrievalProtocol(queries_per_id=2, seed=7).build_query_gallery(data)
    assert model.evaluate(query, gallery, verbose=False).num_queries == 12


def test_explicit_splits_score_a_perfect_match() -> None:
    query = ReIDSplit(image_paths=["q"], pids=np.array([1]), camids=np.array([0]))
    # g1 is the query's person on another camera; the same person on the same camera would not count as a match.
    gallery = ReIDSplit(image_paths=["g1", "g2"], pids=np.array([1, 2]), camids=np.array([1, 2]))
    model = _model_with({"q": [1.0, 0.0], "g1": [1.0, 0.0], "g2": [0.0, 1.0]})

    result = model.evaluate(query, gallery, verbose=False)

    assert result.metrics["map"] == 100.0
    assert result.metrics["rank1"] == 100.0


def test_reassociation_scores_mot_crops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for track_id in (1, 2):
        for frame in (60, 90, 120):
            write_mot_crop(tmp_path, "SEQ-01", track_id, frame)
    model = _model()

    def embeddings(paths: list[str], **kwargs: object) -> np.ndarray:
        """Track 1 and track 2 get opposite embeddings, so every query ranks its own track first."""
        return np.array([[1.0, 0.0] if Path(p).parent.name.endswith("_1") else [0.0, 1.0] for p in paths])

    monkeypatch.setattr(model, "extract_embeddings_from_paths", embeddings)

    result = model.evaluate(
        ReIDDataset(tmp_path), protocol=ReassociationProtocol(gap_frames=30, query_stride=60), verbose=False
    )

    assert result.metrics == {"rank1": 100.0, "mrr": 100.0}
    # Queries start every 60 frames: frame 60 finds both tracks at frame 90, frame 120 has no frame 150.
    assert result.num_queries == 2
    assert result.metadata["num_skipped_queries"] == 2


def test_raises_when_no_query_can_be_scored() -> None:
    """All-zero metrics would be indistinguishable from a model that misses every query."""
    query = ReIDSplit(image_paths=["q"], pids=np.array([7]), camids=np.array([0]))
    gallery = ReIDSplit(image_paths=["g"], pids=np.array([1]), camids=np.array([1]))  # person 7 is not in it
    with pytest.raises(ValueError, match="No query has a correct match"):
        _model_with({"q": [1.0, 0.0], "g": [1.0, 0.0]}).evaluate(query, gallery, verbose=False)


def test_masked_evaluation_keeps_raw_distances() -> None:
    """A protocol's candidate mask restricts scoring; the returned distances stay unmasked."""

    class Masked(RetrievalProtocol):
        def _build_query_gallery(self, data, gallery=None):
            query, gallery, _, metadata = super()._build_query_gallery(data, gallery)
            return query, gallery, np.array([[False, True, True]]), metadata

    query = ReIDSplit(image_paths=["q"], pids=np.array([1]), camids=np.array([0]))
    gallery = ReIDSplit(image_paths=["g1", "g2", "g3"], pids=np.array([1, 2, 1]), camids=np.array([1, 2, 3]))
    model = _model_with({"q": [1.0, 0.0], "g1": [1.0, 0.0], "g2": [0.8, 0.2], "g3": [0.0, 1.0]})

    result = model.evaluate(query, gallery, protocol=Masked(), return_distmat=True, verbose=False)

    # The mask drops g1, the exact match. The ranking left is g2 (another person), then g3 (a match at rank 2).
    assert result.metrics["map"] == pytest.approx(50.0)
    assert result.distmat[0, 0] == pytest.approx(0.0)  # the distance to g1 is still reported


def test_protocol_must_be_an_instance() -> None:
    split = ReIDSplit(image_paths=["a", "b"], pids=np.array([1, 2]), camids=np.array([0, 0]))

    with pytest.raises(TypeError, match="EvaluationProtocol instance"):
        _model().evaluate(split, split, protocol=RetrievalProtocol)  # the class, not RetrievalProtocol()


def test_embeds_shared_paths_once_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    query = ReIDSplit(image_paths=["shared"], pids=np.array([1]), camids=np.array([0]))
    gallery = ReIDSplit(image_paths=["shared", "other"], pids=np.array([1, 2]), camids=np.array([1, 2]))
    calls = []

    def extract(paths: list[str], **kwargs: object) -> np.ndarray:
        calls.append(list(paths))
        return np.array([[1.0, 0.0], [0.0, 1.0]])

    monkeypatch.setattr(model, "extract_embeddings_from_paths", extract)
    model.evaluate(query, gallery, verbose=False)
    assert calls == [["shared", "other"]]


def test_non_finite_embeddings_raise() -> None:
    """A diverged model would otherwise produce plausible-looking metrics from NaN distances."""
    query = ReIDSplit(image_paths=["q"], pids=np.array([1]), camids=np.array([0]))
    gallery = ReIDSplit(image_paths=["g"], pids=np.array([1]), camids=np.array([1]))
    with pytest.raises(ValueError, match="non-finite"):
        _model_with({"q": [np.nan, 0.0], "g": [1.0, 0.0]}).evaluate(query, gallery, verbose=False)
