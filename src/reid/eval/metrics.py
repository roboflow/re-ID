# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""ReID evaluation results, distances, and retrieval metrics (CMC, mAP, mINP)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReIDMetrics:
    """CMC / mAP / mINP scores (percentages) for one evaluation run."""

    mean_average_precision: float
    rank1: float
    rank5: float
    rank10: float
    minp: float
    num_queries: int

    def __str__(self) -> str:
        return (
            f"mAP: {self.mean_average_precision:.1f}%  "
            f"Rank-1: {self.rank1:.1f}%  "
            f"Rank-5: {self.rank5:.1f}%  "
            f"Rank-10: {self.rank10:.1f}%  "
            f"mINP: {self.minp:.1f}%  "
            f"(n={self.num_queries})"
        )


@dataclass
class ReIDResult:
    """What :meth:`reid.ReIDModel.evaluate` returns.

    ``metrics`` maps the protocol's metric names to percentages. ``distmat`` is
    empty unless ``return_distmat=True``.
    """

    metrics: dict[str, float]
    num_queries: int
    distmat: np.ndarray
    metadata: dict


_DISTANCES = ("cosine", "euclidean")
_MAX_RANK = 10


def _validate_distance(distance: object) -> None:
    if distance not in _DISTANCES:
        raise ValueError(f"distance must be 'cosine' or 'euclidean', got {distance!r}.")


def _distance_matrix(query_embeddings: np.ndarray, gallery_embeddings: np.ndarray, metric: str) -> np.ndarray:
    """Build a query x gallery distance matrix (``cosine`` or ``euclidean``)."""
    if metric == "cosine":
        normalized_query = (
            query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-12)
        ).astype(np.float32, copy=False)
        normalized_gallery = (
            gallery_embeddings / (np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-12)
        ).astype(np.float32, copy=False)
        distmat = normalized_query @ normalized_gallery.T
        distmat *= -1.0
        distmat += 1.0
        return distmat
    distmat = (query_embeddings @ gallery_embeddings.T).astype(np.float32, copy=False)
    distmat *= -2.0
    distmat += (query_embeddings**2).sum(axis=1, keepdims=True)
    distmat += (gallery_embeddings**2).sum(axis=1, keepdims=True).T
    np.maximum(distmat, 0.0, out=distmat)
    np.sqrt(distmat, out=distmat)
    return distmat


def _compute_reid_metrics(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    *,
    gallery_junk_pids: frozenset[int] = frozenset({-1}),
    candidate_mask: np.ndarray | None = None,
    exclude_same_pid_camid: bool = True,
) -> ReIDMetrics:
    """Compute CMC, mAP, and mINP from a query x gallery distance matrix.

    Excludes same-(pid, camid) gallery matches by default and gallery junk person IDs
    (``pid in gallery_junk_pids``). Market-1501 gallery distractors use
    ``gallery_junk_pids=frozenset({-1, 0})``; datasets where ``pid=0`` is
    valid should keep the default ``{-1}`` only.

    Args:
        distmat: Distance matrix ``(num_queries, num_gallery)`` (lower = closer).
        q_pids: Query person IDs.
        g_pids: Gallery person IDs.
        q_camids: Query camera IDs.
        g_camids: Gallery camera IDs.
        gallery_junk_pids: Gallery person IDs treated as junk during ranking.
        candidate_mask: Boolean array shaped ``(num_queries, num_gallery)``.
            ``True`` marks an eligible candidate; ``False`` excludes it from
            ranking and relevance counts. Camera and junk-ID filtering still
            apply. Queries without an eligible positive are omitted from the
            averages and ``num_queries``. ``None`` adds no candidate restriction.
        exclude_same_pid_camid: Exclude same-person, same-camera matches.

    Returns:
        ``ReIDMetrics`` with scores as percentages.
    """
    num_q, num_g = distmat.shape

    queries_matched_in_top_k = np.zeros(_MAX_RANK, dtype=np.float64)
    ap_list: list[float] = []
    inp_list: list[float] = []

    junk_pid_array = np.array(sorted(gallery_junk_pids), dtype=g_pids.dtype) if gallery_junk_pids else np.empty(0)

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = np.argsort(distmat[q_idx])
        sorted_g_pids = g_pids[order]
        sorted_g_camids = g_camids[order]

        junk = (
            (sorted_g_pids == q_pid) & (sorted_g_camids == q_camid)
            if exclude_same_pid_camid
            else np.zeros(num_g, dtype=bool)
        )
        if junk_pid_array.size > 0:
            junk |= np.isin(sorted_g_pids, junk_pid_array)
        valid = ~junk
        if candidate_mask is not None:
            valid &= candidate_mask[q_idx, order]

        sorted_g_pids_valid = sorted_g_pids[valid]
        matches = (sorted_g_pids_valid == q_pid).astype(np.float32)

        num_rel = int(matches.sum())
        if num_rel == 0:
            continue

        match_in_top_k = np.minimum(matches.cumsum(), 1.0)
        if len(match_in_top_k) < _MAX_RANK:
            match_in_top_k = np.pad(
                match_in_top_k, (0, _MAX_RANK - len(match_in_top_k)), constant_values=match_in_top_k[-1]
            )
        else:
            match_in_top_k = match_in_top_k[:_MAX_RANK]
        queries_matched_in_top_k += match_in_top_k

        num_valid = len(matches)
        precision_at_k = matches.cumsum() / (np.arange(num_valid) + 1)
        ap = float((precision_at_k * matches).sum() / num_rel)
        ap_list.append(ap)

        last_true = int(np.where(matches)[0][-1]) + 1
        inp_list.append(num_rel / last_true)

    n = len(ap_list)
    if n == 0:
        return ReIDMetrics(
            mean_average_precision=0.0,
            rank1=0.0,
            rank5=0.0,
            rank10=0.0,
            minp=0.0,
            num_queries=0,
        )

    rank_k = (queries_matched_in_top_k / n) * 100.0

    return ReIDMetrics(
        mean_average_precision=float(np.mean(ap_list)) * 100.0,
        rank1=float(rank_k[0]),
        rank5=float(rank_k[4]),
        rank10=float(rank_k[9]),
        minp=float(np.mean(inp_list)) * 100.0,
        num_queries=n,
    )
