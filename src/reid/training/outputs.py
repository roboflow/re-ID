# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""What a training run writes and prints: ``training_config.json``, ``last/``, ``metrics.csv`` and the epoch line."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reid.model import ReIDModel


def write_training_config(output_dir: Path, config: dict[str, object]) -> None:
    """Create ``output_dir`` and write ``config`` to ``training_config.json`` in it.

    Args:
        output_dir: The run's output folder.
        config: The run's settings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # default=str writes values JSON has no type for, such as a function in lr_scheduler_kwargs
    (output_dir / "training_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")


def save_epoch(model: ReIDModel, output_dir: Path, row: dict[str, float]) -> None:
    """Save ``model`` to ``last/`` in ``output_dir`` and append ``row`` to ``metrics.csv``.

    ``model.save`` writes into a temporary folder first, then each file is moved into ``last/`` with ``os.replace``,
    so after a crash each file in ``last/`` is whole, from this epoch or the one before.

    Args:
        model: Model after the epoch.
        output_dir: The run's output folder.
        row: The epoch's history row; its keys are the columns of ``metrics.csv``.
    """
    last = output_dir / "last"
    last.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
        model.save(temporary)
        for name in os.listdir(temporary):
            os.replace(Path(temporary) / name, last / name)

    metrics = output_dir / "metrics.csv"
    is_new = not metrics.exists()
    with metrics.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def format_epoch_line(row: dict[str, float], epochs: int) -> str:
    """Format one epoch of the history as the line printed after it.

    Args:
        row: One epoch of the history.
        epochs: Number of epochs in the run.

    Returns:
        For example ``epoch 6/40  lr 3.0e-04  train/loss 1.90  train/classification_loss 1.52 ...``.
    """
    values = [f"{name} {value:.2f}" for name, value in row.items() if name not in ("epoch", "lr", "time")]
    return "  ".join([f"epoch {row['epoch']}/{epochs}", f"lr {row['lr']:.1e}", *values])
