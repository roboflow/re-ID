# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Writing a training run's ``training_config.json`` and ``metrics.csv``, and its line after each epoch."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def write_training_config(output_dir: Path, config: dict[str, object]) -> None:
    """Create ``output_dir`` and write ``config`` to ``training_config.json`` in it.

    Args:
        output_dir: The run's output folder.
        config: The run's settings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # default=str writes values JSON has no type for, such as a function in lr_scheduler_kwargs
    (output_dir / "training_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")


def append_metrics_row(output_dir: Path, row: dict[str, float], columns: list[str]) -> None:
    """Append ``row`` to ``metrics.csv`` in ``output_dir``, writing ``columns`` as the header first if the file is new.

    Args:
        output_dir: The run's output folder.
        row: One epoch of the history. A column it doesn't hold, such as the losses in the row before training, is
            left blank.
        columns: Every column of the run, in order.
    """
    metrics = output_dir / "metrics.csv"
    is_new = not metrics.exists()
    with metrics.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def format_epoch_line(row: dict[str, float], epochs: int) -> str:
    """Format one epoch of the history as the line printed after it.

    Args:
        row: One epoch of the history.
        epochs: Number of epochs in the run.

    Returns:
        For example ``epoch 6/40  lr 3.0e-04  train/loss 1.90  train/classification_loss 1.52 ...``. The scores
        before training, recorded with ``epoch`` 0, read ``before training  val/map 41.20 ...``.
    """
    # The scores before training have no epoch trained, learning rate or losses
    if row["epoch"] == 0:
        parts = ["before training"]
    else:
        parts = [f"epoch {row['epoch']}/{epochs}", f"lr {row['lr']:.1e}"]
    parts += [f"{name} {value:.2f}" for name, value in row.items() if name not in ("epoch", "lr", "time")]
    return "  ".join(parts)
