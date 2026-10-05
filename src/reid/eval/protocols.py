# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Evaluation protocols: which images are compared and which metrics are reported."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from reid.checks import validate_candidate_mask, validate_positive_int
from reid.data.identity_crops import (
    ReIDDataset,
    _check_query_params,
    _retrieval_split,
)
from reid.data.reassociation import _build_reassociation_candidates
from reid.data.splits import ReIDSplit
from reid.eval.metrics import ReIDMetrics, _validate_distance


class EvaluationProtocol(ABC):
    """How to evaluate: which images are compared and which metrics are reported.

    A protocol holds settings only, so one instance can evaluate any number of
    datasets and models. :meth:`build_query_gallery` picks the images to compare, and must be
    deterministic: the same data and settings select the same images, which is
    what lets several models, or several training epochs, be scored on
    identical inputs.

    Subclasses implement ``_build_query_gallery`` and :meth:`metric_values`.

    Attributes:
        name: Protocol name recorded in result metadata.
        metric_names: Metrics that :meth:`metric_values` reports.
        distance: ``"cosine"`` or ``"euclidean"``, used to compare embeddings.
        exclude_same_pid_camid: Leave out gallery images with the query's person
            and camera ID, as the Market-1501 protocol does.
    """

    name: str
    metric_names: tuple[str, ...]
    distance: str = "cosine"
    exclude_same_pid_camid: bool = True

    def build_query_gallery(
        self,
        data: ReIDDataset | ReIDSplit,
        gallery: ReIDSplit | None = None,
    ) -> tuple[ReIDSplit, ReIDSplit, np.ndarray | None, dict]:
        """Pick the query images, gallery images, and candidates to compare.

        Args:
            data: Identity-folder dataset, or a query split with ``gallery``.
            gallery: Gallery split when ``data`` is the query split.

        Returns:
            ``(query, gallery, candidate_mask, metadata)``. ``candidate_mask``
            limits which gallery images each query is ranked against (``None``
            ranks against all of them); ``metadata`` records the protocol
            settings and counts.
        """
        _validate_distance(self.distance)
        selection = self._build_query_gallery(data, gallery)
        query, gallery_split, candidate_mask, _ = selection
        if not query.image_paths or not gallery_split.image_paths:
            raise ValueError("Selected query and gallery must both be nonempty.")
        if candidate_mask is not None:
            validate_candidate_mask(candidate_mask, len(query), len(gallery_split))
        return selection

    @abstractmethod
    def _build_query_gallery(
        self,
        data: ReIDDataset | ReIDSplit,
        gallery: ReIDSplit | None = None,
    ) -> tuple[ReIDSplit, ReIDSplit, np.ndarray | None, dict]:
        """Build this protocol's query, gallery, candidate mask, and metadata."""

    @abstractmethod
    def metric_values(self, metrics: ReIDMetrics) -> dict[str, float]:
        """Map computed statistics to the metrics this protocol reports."""


class RetrievalProtocol(EvaluationProtocol):
    """Standard retrieval over crop folders or explicit benchmark splits.

    Query sampling options apply only when deriving splits from crop folders.
    On crops cut from video, a query's nearest gallery images are often the
    same person a few frames apart, which inflates retrieval scores; use
    :class:`ReassociationProtocol` to measure matching across a frame gap.

    Args:
        queries_per_id: Query images taken from each identity. Defaults to 1.
        min_crops: Images an identity needs to be evaluated. Defaults to
            ``queries_per_id + 1``, so every query keeps a correct match.
        seed: Seed for choosing each identity's query images. Defaults to 0.
        distance: ``"cosine"`` or ``"euclidean"``.
    """

    name = "retrieval"
    metric_names = ("map", "rank1", "rank5", "rank10", "minp")

    def __init__(
        self,
        *,
        queries_per_id: int | None = None,
        min_crops: int | None = None,
        seed: int | None = None,
        distance: str = "cosine",
    ) -> None:
        if queries_per_id is not None:
            validate_positive_int(queries_per_id, "queries_per_id")
        if min_crops is not None:
            validate_positive_int(min_crops, "min_crops")
        self._sampling_set = any(value is not None for value in (queries_per_id, min_crops, seed))
        self.queries_per_id = 1 if queries_per_id is None else queries_per_id
        self.min_crops = self.queries_per_id + 1 if min_crops is None else min_crops
        self.seed = 0 if seed is None else seed
        self.distance = distance
        _check_query_params(queries_per_id=self.queries_per_id, min_crops=self.min_crops)

    def _build_query_gallery(
        self,
        data: ReIDDataset | ReIDSplit,
        gallery: ReIDSplit | None = None,
    ) -> tuple[ReIDSplit, ReIDSplit, np.ndarray | None, dict]:
        if gallery is not None:
            if not isinstance(data, ReIDSplit) or not isinstance(gallery, ReIDSplit):
                raise TypeError("With an explicit gallery, data must be the query ReIDSplit and gallery a ReIDSplit.")
            if self._sampling_set:
                raise ValueError("Crop derivation settings only apply when the partition is derived.")
            query, gallery_split, source, parameters = data, gallery, "explicit", {}
        else:
            if not isinstance(data, ReIDDataset):
                raise TypeError(f"Derived retrieval needs a ReIDDataset, got {type(data).__name__}.")
            parameters = {"queries_per_id": self.queries_per_id, "min_crops": self.min_crops, "seed": self.seed}
            query, gallery_split = _retrieval_split(data.split, **parameters)
            source = "derived"
        metadata = {
            "protocol": self.name,
            "source": source,
            "parameters": parameters,
            "distance": self.distance,
            "num_query_crops": len(query),
            "num_gallery_crops": len(gallery_split),
        }
        return query, gallery_split, None, metadata

    def metric_values(self, metrics: ReIDMetrics) -> dict[str, float]:
        return {
            "map": metrics.mean_average_precision,
            "rank1": metrics.rank1,
            "rank5": metrics.rank5,
            "rank10": metrics.rank10,
            "minp": metrics.minp,
        }


class ReassociationProtocol(EvaluationProtocol):
    """Rank same-sequence candidates at a fixed frame gap after each query.

    Queries use frames divisible by ``query_stride`` and need a future positive
    among at least two candidates; other queries are skipped.

    Args:
        gap_frames: Frames between a query and its candidates.
        query_stride: Queries are taken at frames divisible by this value.
    """

    name = "reassociation"
    metric_names = ("rank1", "mrr")
    exclude_same_pid_camid = False

    def __init__(self, *, gap_frames: int = 30, query_stride: int = 60) -> None:
        validate_positive_int(gap_frames, "gap_frames")
        validate_positive_int(query_stride, "query_stride")
        self.gap_frames = gap_frames
        self.query_stride = query_stride

    def _build_query_gallery(
        self,
        data: ReIDDataset | ReIDSplit,
        gallery: ReIDSplit | None = None,
    ) -> tuple[ReIDSplit, ReIDSplit, np.ndarray | None, dict]:
        if gallery is not None:
            raise ValueError("ReassociationProtocol constructs its own gallery; gallery must be omitted.")
        if not isinstance(data, ReIDDataset):
            raise TypeError(f"Re-association needs a ReIDDataset of MOT crops, got {type(data).__name__}.")
        query, gallery_split, mask, counts = _build_reassociation_candidates(
            data.split, gap_frames=self.gap_frames, query_stride=self.query_stride
        )
        skipped = (
            counts["skipped_missing_future"]
            + counts["skipped_absent_positive"]
            + counts["skipped_insufficient_candidates"]
        )
        metadata = {
            "protocol": self.name,
            "parameters": {"gap_frames": self.gap_frames, "query_stride": self.query_stride},
            "distance": self.distance,
            "num_skipped_queries": skipped,
            **counts,
        }
        return query, gallery_split, mask, metadata

    def metric_values(self, metrics: ReIDMetrics) -> dict[str, float]:
        # Each scored query has exactly one positive, so AP equals reciprocal rank.
        return {"rank1": metrics.rank1, "mrr": metrics.mean_average_precision}
