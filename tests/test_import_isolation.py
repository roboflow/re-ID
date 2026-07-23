# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Package import isolation for the optional training submodule."""

from __future__ import annotations

import subprocess
import sys


def test_import_reid_does_not_load_training() -> None:
    """``import reid`` must not pull the torch training stack."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import reid; "
                "assert 'reid.training' not in sys.modules; "
                "assert 'reid.training.trainer' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
