# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Load and partition per-identity crop folders into ``ReIDSplit`` objects.

The canonical flow is ``source format -> loader -> ReIDSplit -> optional
partitioning -> Dataset/evaluator``. :class:`ReIDDataset` turns an
identity-folder layout (``<root>/<identity>/<crop>.jpg``) into a single
``ReIDSplit``. The split builders below then cut a split into query/gallery
(and optional train) splits; they never touch the filesystem.

Query and gallery crops receive synthetic ``camid`` 0/1 so single-camera MOT
crops still form valid cross-camera retrieval pairs under the gallery-eval junk
rule.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from reid.checks import validate_positive_int
from reid.data.splits import ReIDSplit

# Image suffixes supported by the identity-folder loaders.
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

_QUERY_CAMID = 0
_GALLERY_CAMID = 1


def _iter_identity_crops(root: Path) -> Iterator[list[Path]]:
    """Yield sorted image files per identity folder, skipping empty folders."""
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        images = sorted(
            path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if images:
            yield images


def _load_identity_crops(root: str | Path) -> ReIDSplit:
    """Scan an identity-folder crop dataset into a single :class:`ReIDSplit`.

    Expects ``<root>/<identity>/<crop>.(jpg|jpeg|png|bmp)``. Identity folders are
    sorted by name and assigned contiguous person IDs (``0..N-1``). Camera IDs
    are set to ``-1`` because on-disk crops carry no camera information; the
    split builders assign synthetic camera IDs where retrieval needs them.

    Args:
        root: Dataset root with one subdirectory per identity.

    Returns:
        A :class:`ReIDSplit` over every crop found, with contiguous ``pids`` and
        ``camids`` all equal to ``-1``.

    Raises:
        FileNotFoundError: If ``root`` is not a directory, or if no crops are
            found.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Crop dataset root not found: {root}")

    image_paths: list[str] = []
    pids: list[int] = []
    for pid, images in enumerate(_iter_identity_crops(root)):
        image_paths.extend(str(path) for path in images)
        pids.extend([pid] * len(images))

    if not image_paths:
        raise FileNotFoundError(f"No identity crops found under {root}. Expected <root>/<identity>/<crop>.jpg.")

    return ReIDSplit(
        image_paths=image_paths,
        pids=np.array(pids, dtype=np.int32),
        camids=np.full(len(image_paths), -1, dtype=np.int32),
    )


def _retrieval_split(
    split: ReIDSplit,
    *,
    queries_per_id: int,
    min_crops: int,
    seed: int,
) -> tuple[ReIDSplit, ReIDSplit]:
    """Partition a crop split into query and gallery splits for retrieval eval.

    Every identity with at least ``min_crops`` crops contributes ``queries_per_id``
    crops to the query split (synthetic ``camid`` 0) and the remainder to the
    gallery split (synthetic ``camid`` 1). Crops are shuffled per identity with
    ``seed`` before the query/gallery cut so query selection is deterministic but
    not tied to filename order.

    Args:
        split: A crop split, typically ``ReIDDataset(root).split``.
        queries_per_id: Query crops taken from each eligible identity.
        min_crops: Minimum crops required for an identity to be eligible.
        seed: Shuffle seed for per-identity query selection.

    Returns:
        ``(query, gallery)`` splits.

    ``RetrievalProtocol`` validates ``queries_per_id`` and ``min_crops`` when it
    is built, so they are not checked again here.

    Raises:
        ValueError: If no identity has ``min_crops`` crops.
    """
    indices_by_pid = _indices_by_label(split.pids.tolist())
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


def _check_query_params(*, queries_per_id: int, min_crops: int) -> None:
    """Reject incompatible ``queries_per_id`` / ``min_crops`` combinations.

    An eligible identity must yield at least one query and one gallery crop, so
    ``min_crops`` has to exceed ``queries_per_id``. The combination is rejected
    explicitly rather than silently widened.
    """
    validate_positive_int(queries_per_id, "queries_per_id")
    if min_crops <= queries_per_id:
        raise ValueError(
            f"min_crops must be > queries_per_id to leave gallery crops, "
            f"got min_crops={min_crops}, queries_per_id={queries_per_id}."
        )


def _indices_by_label(labels: Sequence[int]) -> dict[int, list[int]]:
    """Map each identity label to the positions of its crops, in input order.

    Used by the retrieval split builder, which groups raw person IDs.

    Args:
        labels: Identity label per crop, in crop order.

    Returns:
        Label → list of crop positions, preserving input order within each label.
    """
    grouped: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        grouped.setdefault(int(label), []).append(index)
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


class ReIDDataset:
    """An identity-folder image dataset, scanned once.

    An index of identity-labelled images, not a ``torch.utils.data.Dataset``: pass
    it to :meth:`reid.ReIDModel.evaluate`, or read ``split`` for its image paths
    and person IDs.

    Images may be tracking crops or any identity-labelled pictures; only the
    ``<root>/<identity>/<image>`` layout matters. The filesystem scan happens once,
    in the constructor, so hold one instance and reuse it; evaluation protocols
    select query and gallery images from it.

    Benchmark datasets are not covered: Market-1501 and MSMT17 ship their own
    query/gallery definitions, so load those with
    :func:`~reid.data.load_market1501` / :func:`~reid.data.load_msmt17` and pass
    the two splits explicitly rather than re-deriving them here.

    Args:
        root: Dataset root holding one folder per identity.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._split = _load_identity_crops(self._root)

    @property
    def root(self) -> Path:
        """Folder the dataset was scanned from."""
        return self._root

    @property
    def split(self) -> ReIDSplit:
        """Every image found, with its person ID and camera ID."""
        return self._split

    def __len__(self) -> int:
        return len(self.split)

    @property
    def num_identities(self) -> int:
        """Distinct identities in the folder."""
        return len(set(self.split.pids.tolist()))

    def __repr__(self) -> str:
        return f"ReIDDataset({str(self.root)!r}, images={len(self)}, identities={self.num_identities})"
