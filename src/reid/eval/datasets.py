# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Backward-compatible re-exports; prefer :mod:`reid.data`."""

from __future__ import annotations

from reid.data.benchmarks import (
    load_market1501 as load_market1501,
)
from reid.data.benchmarks import (
    load_msmt17 as load_msmt17,
)
from reid.data.splits import (
    MARKET1501_GALLERY_JUNK_PIDS as MARKET1501_GALLERY_JUNK_PIDS,
)
from reid.data.splits import ReIDSplit as ReIDSplit
