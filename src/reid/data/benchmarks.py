# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""ReID benchmark dataset loaders (Market-1501, MSMT17)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from deprecate import deprecated, void

from reid.data.splits import ReIDSplit

# --------------------------------------------------------------------------- #
# MSMT17
# --------------------------------------------------------------------------- #


def _parse_msmt17_camid(filename: str) -> int:
    """Extract 0-indexed camid from an MSMT17 image filename.

    Filename format: ``<pid>_<idx>_<camid>_<scene>_<frame>_<track>.jpg``, with a
    1-indexed camera field. Example: ``0001_019_07_0303morning_0020_1.jpg`` → camid = 6.
    """
    return int(Path(filename).stem.split("_")[2]) - 1


def _load_msmt17_list(root: str | Path, list_name: str, image_dir: str) -> ReIDSplit:
    """Load the images the list file ``root/list_name`` names, read against ``root/image_dir``."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"MSMT17 root not found: {root}")
    image_root = root / image_dir
    if not image_root.is_dir():
        raise FileNotFoundError(f"MSMT17 sub-directory not found: {image_root}")
    list_file = root / list_name
    if not list_file.exists():
        raise FileNotFoundError(f"MSMT17 list file not found: {list_file}")
    paths, pids, camids = [], [], []
    for line_number, line in enumerate(list_file.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"{list_file}:{line_number}: expected 'path pid', got {line.strip()!r}.")
        rel_path, pid = parts
        paths.append(str(image_root / rel_path))
        pids.append(int(pid))
        camids.append(_parse_msmt17_camid(Path(rel_path).name))
    return ReIDSplit(
        image_paths=paths,
        pids=np.array(pids, dtype=np.int32),
        camids=np.array(camids, dtype=np.int32),
        gallery_junk_pids=frozenset({-1}),
    )


def load_msmt17_test(root: str | Path) -> tuple[ReIDSplit, ReIDSplit]:
    """Load the MSMT17 test query and gallery from ``root``.

    Expects ``test/``, ``list_query.txt``, and ``list_gallery.txt``, whose lines
    read ``path pid``. The camera ID comes from the image file name.

    Args:
        root: Path to the MSMT17 directory.

    Returns:
        Query and gallery as ``(ReIDSplit, ReIDSplit)``.

    Raises:
        FileNotFoundError: If ``root``, ``test/``, ``list_query.txt``, or
            ``list_gallery.txt`` does not exist.
        ValueError: If a list line is not ``path pid``.
    """
    query = _load_msmt17_list(root, "list_query.txt", "test")
    gallery = _load_msmt17_list(root, "list_gallery.txt", "test")
    return query, gallery


def load_msmt17_train(root: str | Path) -> ReIDSplit:
    """Load the MSMT17 training images from ``root``.

    Expects ``train/`` and ``list_train.txt``, whose lines read ``path pid``.
    The camera ID comes from the image file name. The images in
    ``list_val.txt`` are left out, as torchreid and fast-reid do: load them
    with ``load_msmt17_val``.

    Args:
        root: Path to the MSMT17 directory.

    Returns:
        The training images as one ``ReIDSplit``.

    Raises:
        FileNotFoundError: If ``root``, ``train/``, or ``list_train.txt`` does
            not exist.
        ValueError: If a list line is not ``path pid``.
    """
    return _load_msmt17_list(root, "list_train.txt", "train")


def load_msmt17_val(root: str | Path) -> ReIDSplit:
    """Load the MSMT17 validation images from ``root``.

    Expects ``train/`` and ``list_val.txt``, whose lines read ``path pid``. The
    camera ID comes from the image file name.

    ``list_val.txt`` holds other images of the training people. A model trained
    on ``load_msmt17_train`` has seen these people, so this split does not test
    it on new people.

    Args:
        root: Path to the MSMT17 directory.

    Returns:
        The validation images as one ``ReIDSplit``.

    Raises:
        FileNotFoundError: If ``root``, ``train/``, or ``list_val.txt`` does
            not exist.
        ValueError: If a list line is not ``path pid``.
    """
    return _load_msmt17_list(root, "list_val.txt", "train")


@deprecated(target=load_msmt17_test, deprecated_in="0.2.0", remove_in="0.4.0")
def load_msmt17(root: str | Path) -> tuple[ReIDSplit, ReIDSplit]:
    """Load the MSMT17 test query and gallery from ``root``.

    Deprecated since 0.2.0 and removed in 0.4.0: use ``load_msmt17_test``,
    which takes the same ``root`` and returns the same query and gallery.

    Args:
        root: Path to the MSMT17 directory.

    Returns:
        Query and gallery as ``(ReIDSplit, ReIDSplit)``.
    """
    return void(root)


# --------------------------------------------------------------------------- #
# Market-1501
# --------------------------------------------------------------------------- #


def _parse_market_filename(filename: str) -> tuple[int, int]:
    """Parse ``(pid, camid)`` from a Market-1501 filename.

    ``-1_*`` → junk (``pid=-1``), ignored when ranking; ``0000_*`` → distractor
    (``pid=0``), kept as a wrong candidate, as torchreid's ``market1501.py`` does.
    """
    parts = Path(filename).stem.split("_")
    try:
        pid = int(parts[0])
        camid = int(parts[1][1]) - 1  # "c1" → 0
    except (IndexError, ValueError):
        raise ValueError(
            f"Market-1501 file name {Path(filename).name!r} does not follow '<person>_c<camera>...jpg'."
        ) from None
    return pid, camid


def _load_market1501_dir(root: str | Path, image_dir: str) -> ReIDSplit:
    """Load the Market-1501 image directory ``root/image_dir``."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Market-1501 root not found: {root}")
    subdir = root / image_dir
    if not subdir.exists():
        raise FileNotFoundError(f"Market-1501 sub-directory not found: {subdir}")
    paths, pids, camids = [], [], []
    for img_path in sorted(subdir.glob("*.jpg")):
        pid, camid = _parse_market_filename(img_path.name)
        paths.append(str(img_path))
        pids.append(pid)
        camids.append(camid)
    return ReIDSplit(
        image_paths=paths,
        pids=np.array(pids, dtype=np.int32),
        camids=np.array(camids, dtype=np.int32),
        gallery_junk_pids=frozenset({-1}),
    )


def load_market1501_test(root: str | Path) -> tuple[ReIDSplit, ReIDSplit]:
    """Load the Market-1501 test query and gallery from ``root``.

    Expects ``query/`` and ``bounding_box_test/``. Person and camera IDs are
    parsed from filenames.

    Args:
        root: Path to the Market-1501 directory.

    Returns:
        Query and gallery as ``(ReIDSplit, ReIDSplit)``.

    Raises:
        FileNotFoundError: If ``root``, ``query/``, or ``bounding_box_test/``
            does not exist.
        ValueError: If an image file name does not start with
            ``<person>_c<camera>``, such as a hidden ``._*`` file macOS writes.
    """
    query = _load_market1501_dir(root, "query")
    gallery = _load_market1501_dir(root, "bounding_box_test")
    return query, gallery


def load_market1501_train(root: str | Path) -> ReIDSplit:
    """Load the Market-1501 training images from ``root``.

    Expects ``bounding_box_train/``. Person and camera IDs are parsed from
    filenames. Market-1501 has no validation split.

    Args:
        root: Path to the Market-1501 directory.

    Returns:
        The training images as one ``ReIDSplit``.

    Raises:
        FileNotFoundError: If ``root`` or ``bounding_box_train/`` does not
            exist.
        ValueError: If an image file name does not start with
            ``<person>_c<camera>``, such as a hidden ``._*`` file macOS writes.
    """
    return _load_market1501_dir(root, "bounding_box_train")


@deprecated(target=load_market1501_test, deprecated_in="0.2.0", remove_in="0.4.0")
def load_market1501(root: str | Path) -> tuple[ReIDSplit, ReIDSplit]:
    """Load the Market-1501 test query and gallery from ``root``.

    Deprecated since 0.2.0 and removed in 0.4.0: use ``load_market1501_test``,
    which takes the same ``root`` and returns the same query and gallery.

    Args:
        root: Path to the Market-1501 directory.

    Returns:
        Query and gallery as ``(ReIDSplit, ReIDSplit)``.
    """
    return void(root)
