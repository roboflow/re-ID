# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Validate package distribution versions before tagged release uploads.

Purpose:
    Reject release artifacts whose versions differ from the triggering tag.
Scope:
    Parse wheel and source archive filenames in the build job's ``dist`` directory.
Usage:
    Set ``RELEASE_TAG`` and run this file from the repository root with Python.
Outputs:
    Print a success message when every built distribution matches the tag.
Failure:
    Exit nonzero for a missing tag, empty artifact directory, unsupported artifact,
    invalid version, or any mismatch between a distribution and the release tag.
Used by:
    The tagged build step in ``.github/workflows/build-package.yml`` before uploading.
"""

import os
from pathlib import Path

from packaging.utils import parse_sdist_filename, parse_wheel_filename
from packaging.version import Version


def main() -> None:
    """Fail unless all wheel and source archive versions match the release tag."""
    tag = os.environ["RELEASE_TAG"]
    expected = Version(tag[1:] if tag.startswith("v") else tag)
    distributions = sorted(path for path in Path("dist").iterdir() if path.name != ".gitignore")
    if not distributions:
        raise SystemExit("No distribution artifacts were built")

    versions = set()
    for distribution in distributions:
        if distribution.name.endswith(".whl"):
            _, version, _, _ = parse_wheel_filename(distribution.name)
        elif distribution.name.endswith(".tar.gz"):
            _, version = parse_sdist_filename(distribution.name)
        else:
            raise SystemExit(f"Unexpected distribution artifact: {distribution.name}")
        versions.add(version)

    if versions != {expected}:
        raise SystemExit(
            f"Release tag {expected} does not match built distribution versions {sorted(map(str, versions))}"
        )
    print(f"All distributions match release tag {expected}")


if __name__ == "__main__":
    main()
