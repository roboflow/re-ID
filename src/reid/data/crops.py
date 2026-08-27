# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""A crop folder that can be trained on or evaluated against."""

from __future__ import annotations

from pathlib import Path

from reid.data.identity_crops import (
    build_identity_crop_holdout_split,
    build_identity_crop_retrieval_split,
    load_identity_crops,
)
from reid.data.splits import ReIDSplit


class ReIDCrops:
    """A ``<root>/<identity>/<crop>.jpg`` folder, scanned once.

    The filesystem scan happens in the constructor; the partitions below are
    derived from it on demand and cost microseconds, so hold one instance and
    ask it for whichever view a step needs.

    Benchmark datasets are not covered: Market-1501 and MSMT17 ship their own
    query/gallery definitions, so load those with
    :func:`~reid.data.load_market1501` / :func:`~reid.data.load_msmt17` and pass
    the two splits explicitly rather than re-deriving them here.

    Args:
        root: Crop dataset root.
        identities: Optional whitelist of identity folder names.

    Attributes:
        split: The underlying :class:`ReIDSplit` over every crop found.
    """

    def __init__(self, root: str | Path, *, identities: list[str] | None = None) -> None:
        self.root = Path(root)
        self.split = load_identity_crops(self.root, identities=identities)

    def __len__(self) -> int:
        return len(self.split)

    @property
    def num_identities(self) -> int:
        """Distinct identities in the folder."""
        return len(set(self.split.pids.tolist()))

    def for_retrieval(
        self,
        *,
        queries_per_id: int = 1,
        min_crops: int = 2,
        seed: int = 0,
    ) -> tuple[ReIDSplit, ReIDSplit]:
        """Partition into ``(query, gallery)`` for retrieval evaluation."""
        return build_identity_crop_retrieval_split(
            self.split,
            queries_per_id=queries_per_id,
            min_crops=min_crops,
            seed=seed,
        )

    def for_holdout(
        self,
        *,
        holdout_fraction: float = 0.2,
        queries_per_id: int = 1,
        min_crops: int = 2,
        seed: int = 0,
    ) -> tuple[ReIDSplit, ReIDSplit, ReIDSplit]:
        """Hold whole identities out for retrieval; return ``(train, query, gallery)``."""
        return build_identity_crop_holdout_split(
            self.split,
            holdout_fraction=holdout_fraction,
            queries_per_id=queries_per_id,
            min_crops=min_crops,
            seed=seed,
        )

    def __repr__(self) -> str:
        return f"ReIDCrops({str(self.root)!r}, crops={len(self)}, identities={self.num_identities})"
