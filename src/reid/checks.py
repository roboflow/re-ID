# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Evaluation input checks shared by :mod:`reid.data` and :mod:`reid.eval`.

Internal building block; not part of the curated public API surface.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MAX_PAIRS = 100_000_000


def validate_positive_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def validate_pair_budget(num_queries: int, num_gallery: int, max_pairs: int) -> None:
    """Reject an oversized query/gallery matrix before allocating it."""
    pairs = num_queries * num_gallery
    if pairs > max_pairs:
        raise ValueError(
            f"Evaluation needs a {num_queries:,} x {num_gallery:,} query-by-gallery matrix ({pairs:,} pairs, "
            f"about {pairs * 4 / 1e9:.1f} GB as float32), above max_pairs={max_pairs:,}. "
            "Pass a larger max_pairs if you have the memory."
        )


def validate_candidate_mask(mask: object, num_queries: int, num_gallery: int) -> None:
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != (num_queries, num_gallery):
        raise ValueError("candidate_mask must be a boolean query x gallery array.")
