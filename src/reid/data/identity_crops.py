# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Load and partition per-identity crop folders into ``ReIDSplit`` objects.

The canonical flow is ``source format -> loader -> ReIDSplit -> optional
partitioning -> Dataset/evaluator``. :func:`load_identity_crops` turns an
identity-folder layout (``<root>/<identity>/<crop>.jpg``) into a single
``ReIDSplit``. The ``build_identity_crop_*`` partitioners then consume that split to
produce query/gallery (and optional train) splits; they never touch the
filesystem.

Query and gallery crops receive synthetic ``camid`` 0/1 so single-camera MOT
crops still form valid cross-camera retrieval pairs under the gallery-eval junk
rule.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from reid.data.splits import ReIDSplit

# Image suffixes scanned inside each identity folder. Kept private to this
# module; there is no shared ``constants`` module by design.
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

_QUERY_CAMID = 0
_GALLERY_CAMID = 1


def identity_name_from_crop_path(path: str | Path) -> str:
    """Identity folder name for the ``<root>/<identity>/<crop>.jpg`` layout."""
    return Path(path).parent.name


def load_identity_crops(
    root: str | Path,
    *,
    identities: list[str] | None = None,
) -> ReIDSplit:
    """Scan an identity-folder crop dataset into a single :class:`ReIDSplit`.

    Expects ``<root>/<identity>/<crop>.(jpg|jpeg|png|bmp)``. Identity folders are
    sorted by name and assigned contiguous person IDs (``0..N-1``). Camera IDs
    are set to ``-1`` because on-disk crops carry no camera information; the
    ``build_identity_crop_*`` partitioners assign synthetic camera IDs where retrieval
    needs them.

    Args:
        root: Dataset root with one subdirectory per identity.
        identities: Optional whitelist of identity folder names. Folders outside
            the whitelist are skipped before person IDs are assigned.

    Returns:
        A :class:`ReIDSplit` over every crop found, with contiguous ``pids`` and
        ``camids`` all equal to ``-1``.

    Raises:
        FileNotFoundError: If ``root`` is not a directory, or if no crops are
            found (after applying ``identities``).
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Crop dataset root not found: {root}")

    allowed = set(identities) if identities is not None else None

    image_paths: list[str] = []
    pids: list[int] = []
    next_pid = 0
    for identity_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if allowed is not None and identity_dir.name not in allowed:
            continue
        images = sorted(p for p in identity_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)
        if not images:
            continue
        for image_path in images:
            image_paths.append(str(image_path))
            pids.append(next_pid)
        next_pid += 1

    if not image_paths:
        raise FileNotFoundError(f"No identity crops found under {root}. Expected <root>/<identity>/<crop>.jpg.")

    return ReIDSplit(
        image_paths=image_paths,
        pids=np.array(pids, dtype=np.int32),
        camids=np.full(len(image_paths), -1, dtype=np.int32),
    )


def build_identity_crop_retrieval_split(
    split: ReIDSplit,
    *,
    queries_per_id: int = 1,
    min_crops: int = 2,
    seed: int = 0,
) -> tuple[ReIDSplit, ReIDSplit]:
    """Partition a crop split into query and gallery splits for retrieval eval.

    Every identity with at least ``min_crops`` crops contributes ``queries_per_id``
    crops to the query split (synthetic ``camid`` 0) and the remainder to the
    gallery split (synthetic ``camid`` 1). Crops are shuffled per identity with
    ``seed`` before the query/gallery cut so query selection is deterministic but
    not tied to filename order.

    Args:
        split: A crop split, typically from :func:`load_identity_crops`.
        queries_per_id: Query crops taken from each eligible identity.
        min_crops: Minimum crops required for an identity to be eligible.
        seed: Shuffle seed for per-identity query selection.

    Returns:
        ``(query, gallery)`` splits.

    Raises:
        ValueError: If ``min_crops``/``queries_per_id`` are incompatible (see
            :func:`_check_query_params`), or if no identity is eligible.
    """
    _check_query_params(queries_per_id=queries_per_id, min_crops=min_crops)

    indices_by_pid = _indices_by_pid(split)
    eligible_pids = sorted(pid for pid, indices in indices_by_pid.items() if len(indices) >= min_crops)
    if not eligible_pids:
        raise ValueError(f"No identity has >= {min_crops} crops; cannot build a retrieval split.")

    return _query_gallery_from_pids(
        split,
        indices_by_pid,
        eligible_pids,
        queries_per_id=queries_per_id,
        seed=seed,
    )


def build_identity_crop_holdout_split(
    split: ReIDSplit,
    *,
    holdout_fraction: float = 0.2,
    queries_per_id: int = 1,
    min_crops: int = 2,
    seed: int = 0,
) -> tuple[ReIDSplit, ReIDSplit, ReIDSplit]:
    """Hold out whole identities for retrieval; return train, query, gallery.

    A fraction of the identities that have at least ``min_crops`` crops is drawn
    (deterministically via ``seed``) and turned into query/gallery splits exactly
    as in :func:`build_identity_crop_retrieval_split`. Every remaining crop, including
    crops of identities too small to be eligible for holdout, forms the train
    split with its original camera IDs preserved. The train and evaluation splits
    are identity-disjoint, so retrieval never scores an identity the model saw
    during training.

    Args:
        split: A crop split, typically from :func:`load_identity_crops`.
        holdout_fraction: Fraction of eligible identities reserved for retrieval.
        queries_per_id: Query crops per held-out identity.
        min_crops: Minimum crops required for an identity to be eligible for
            holdout.
        seed: Shuffle seed for the holdout draw and per-identity query selection.

    Returns:
        ``(train, query, gallery)`` splits.

    Raises:
        ValueError: If ``holdout_fraction`` is outside ``(0, 1]``, if
            ``min_crops``/``queries_per_id`` are incompatible, or if no identity
            is eligible for holdout.
    """
    _check_query_params(queries_per_id=queries_per_id, min_crops=min_crops)
    if not 0.0 < holdout_fraction <= 1.0:
        raise ValueError(f"holdout_fraction must be in (0, 1], got {holdout_fraction}.")

    indices_by_pid = _indices_by_pid(split)
    eligible_pids = sorted(pid for pid, indices in indices_by_pid.items() if len(indices) >= min_crops)
    if not eligible_pids:
        raise ValueError(f"No identity has >= {min_crops} crops; cannot build a retrieval holdout.")

    shuffled = eligible_pids.copy()
    random.Random(seed).shuffle(shuffled)
    num_holdout = max(1, round(len(shuffled) * holdout_fraction))
    holdout_pids = set(shuffled[:num_holdout])

    train_indices = [index for index, pid in enumerate(split.pids.tolist()) if pid not in holdout_pids]
    train = _subset_split(split, train_indices)
    query, gallery = _query_gallery_from_pids(
        split,
        indices_by_pid,
        sorted(holdout_pids),
        queries_per_id=queries_per_id,
        seed=seed,
    )
    return train, query, gallery


def _check_query_params(*, queries_per_id: int, min_crops: int) -> None:
    """Reject incompatible ``queries_per_id`` / ``min_crops`` combinations.

    An eligible identity must yield at least one query and one gallery crop, so
    ``min_crops`` has to exceed ``queries_per_id``. The combination is rejected
    explicitly rather than silently widened.
    """
    if queries_per_id < 1:
        raise ValueError(f"queries_per_id must be >= 1, got {queries_per_id}.")
    if min_crops <= queries_per_id:
        raise ValueError(
            f"min_crops must be > queries_per_id to leave gallery crops, "
            f"got min_crops={min_crops}, queries_per_id={queries_per_id}."
        )


def _indices_by_pid(split: ReIDSplit) -> dict[int, list[int]]:
    """Group crop indices by person ID, preserving the split's order."""
    grouped: dict[int, list[int]] = {}
    for index, pid in enumerate(split.pids.tolist()):
        grouped.setdefault(pid, []).append(index)
    return grouped


def _query_gallery_from_pids(
    split: ReIDSplit,
    indices_by_pid: dict[int, list[int]],
    pids: list[int],
    *,
    queries_per_id: int,
    seed: int,
) -> tuple[ReIDSplit, ReIDSplit]:
    """Cut each identity in ``pids`` into query and gallery crops."""
    rng = random.Random(seed)
    query_indices: list[int] = []
    gallery_indices: list[int] = []
    for pid in pids:
        indices = indices_by_pid[pid].copy()
        rng.shuffle(indices)
        query_indices.extend(indices[:queries_per_id])
        gallery_indices.extend(indices[queries_per_id:])

    query = _subset_split(split, query_indices, camid=_QUERY_CAMID)
    gallery = _subset_split(split, gallery_indices, camid=_GALLERY_CAMID)
    return query, gallery


def _subset_split(
    split: ReIDSplit,
    indices: list[int],
    *,
    camid: int | None = None,
) -> ReIDSplit:
    """Select ``indices`` from ``split`` into a new :class:`ReIDSplit`.

    When ``camid`` is given, every crop in the subset is stamped with that
    synthetic camera ID and the default junk rule; otherwise the original camera
    IDs and junk rule are preserved.
    """
    image_paths = [split.image_paths[index] for index in indices]
    pids = np.array([int(split.pids[index]) for index in indices], dtype=np.int32)
    if camid is None:
        camids = np.array([int(split.camids[index]) for index in indices], dtype=np.int32)
        return ReIDSplit(
            image_paths=image_paths,
            pids=pids,
            camids=camids,
            gallery_junk_pids=split.gallery_junk_pids,
        )
    return ReIDSplit(
        image_paths=image_paths,
        pids=pids,
        camids=np.full(len(indices), camid, dtype=np.int32),
    )
