# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Build same-sequence, fixed-frame-gap candidate sets from MOT crops."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from reid.data.mot import _parse_crop_path
from reid.data.splits import ReIDSplit


def _crops_by_frame(split: ReIDSplit) -> dict[str, dict[int, dict[int, Path]]]:
    """Index MOT crops as ``{sequence: {frame: {track_id: path}}}``."""
    frames: dict[str, dict[int, dict[int, Path]]] = defaultdict(lambda: defaultdict(dict))
    for path in map(Path, split.image_paths):
        sequence, track_id, frame = _parse_crop_path(path)
        if track_id in frames[sequence][frame]:
            raise ValueError(f"Duplicate sequence/identity/frame crop: {path}")
        frames[sequence][frame][track_id] = path
    return frames


def _build_reassociation_candidates(
    data: ReIDSplit,
    gallery: ReIDSplit | None = None,
    *,
    gap_frames: int,
    query_stride: int,
) -> tuple[ReIDSplit, ReIDSplit, np.ndarray, dict]:
    """Return the queries, the gallery crops they are ranked against, the candidate mask and counts.

    Queries are the crops of ``data`` at frames divisible by ``query_stride``.
    Candidates are all identities in the same sequence exactly ``gap_frames``
    later, taken from ``gallery``, or from ``data`` when it is omitted. A query
    is scored only if its identity is present among at least two candidates.
    """
    query_frames = _crops_by_frame(data)
    gallery_frames = query_frames if gallery is None else _crops_by_frame(gallery)
    identities = {
        (sequence, identity)
        for frames in (query_frames, gallery_frames)
        for sequence, seq_frames in frames.items()
        for crops in seq_frames.values()
        for identity in crops
    }
    pid_by_identity = {identity: index for index, identity in enumerate(sorted(identities))}

    records: list[tuple[str, int, Path, tuple[tuple[int, Path], ...]]] = []
    skipped = {"skipped_missing_future": 0, "skipped_absent_positive": 0, "skipped_insufficient_candidates": 0}
    # filter queries that dont have possible retrievals
    for sequence, seq_frames in sorted(query_frames.items()):
        for frame, crops in sorted(seq_frames.items()):
            if frame % query_stride:
                continue
            future = gallery_frames.get(sequence, {}).get(frame + gap_frames)
            for identity, query_path in sorted(crops.items()):
                if future is None:
                    skipped["skipped_missing_future"] += 1
                elif len(future) < 2:
                    skipped["skipped_insufficient_candidates"] += 1
                elif identity not in future:
                    skipped["skipped_absent_positive"] += 1
                else:
                    records.append((sequence, identity, query_path, tuple(sorted(future.items()))))

    if not records:
        raise ValueError("No re-association queries have a positive and at least two candidates.")

    gallery_records = sorted(
        {(sequence, identity, path) for sequence, _, _, candidates in records for identity, path in candidates}
    )

    gallery_index = {record: index for index, record in enumerate(gallery_records)}
    # TODO: Store per-query candidate indices and score only eligible pairs to avoid
    # dense query x gallery masks/distances while preserving metric semantics.

    # mark the galleries that associate to which queries
    candidate_mask = np.zeros((len(records), len(gallery_records)), dtype=bool)
    for row, (sequence, _, _, candidates) in enumerate(records):
        for identity, path in candidates:
            candidate_mask[row, gallery_index[(sequence, identity, path)]] = True

    query_split = ReIDSplit(
        image_paths=[str(path) for _, _, path, _ in records],
        pids=np.array([pid_by_identity[(sequence, identity)] for sequence, identity, _, _ in records], dtype=np.int32),
        camids=np.zeros(len(records), dtype=np.int32),
    )
    gallery_split = ReIDSplit(
        image_paths=[str(path) for _, _, path in gallery_records],
        pids=np.array(
            [pid_by_identity[(sequence, identity)] for sequence, identity, _ in gallery_records], dtype=np.int32
        ),
        camids=np.zeros(len(gallery_records), dtype=np.int32),
    )
    candidate_counts = candidate_mask.sum(axis=1)
    metadata = {
        "num_query_crops": len(query_split),
        "num_gallery_crops": len(gallery_split),
        "mean_candidates": float(candidate_counts.mean()),
        "min_candidates": int(candidate_counts.min()),
        "max_candidates": int(candidate_counts.max()),
        **skipped,
    }
    return query_split, gallery_split, candidate_mask, metadata
