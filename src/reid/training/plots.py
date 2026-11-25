# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Plots of a training run, its curves by epoch and the embeddings of a few validation people after each epoch."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from reid.data import ReIDSplit
from reid.eval.metrics import _l2_normalize
from reid.training.outputs import read_metrics_csv


def plot_training_history(
    history: list[dict[str, float]] | str | Path, checkpoint_metric: str | None = "auto"
) -> Figure:
    """Plot a training run's losses, learning rate and validation scores by epoch, and return the figure.

    The figure stacks a panel of the losses, a panel of the validation metrics for a run with validation, starting
    from the scores before training, and a panel of the learning rate on a log scale, on one epoch axis. A dashed line
    on the losses and validation panels marks the epoch ``train`` saves to ``best/``. In a notebook it shows under the
    cell.

    Args:
        history: The list ``ReIDModel.train`` returns, or the ``output_dir`` of a run, whose ``metrics.csv`` is read.
        checkpoint_metric: Metric whose best epoch is marked. ``"auto"`` takes the one the run's
            ``training_config.json`` names when ``history`` is an ``output_dir``, and otherwise ``"mrr"`` or ``"map"``,
            whichever the history holds, as ``train`` picks for re-association and retrieval. ``None`` marks nothing.

    Returns:
        A matplotlib ``Figure``.

    Raises:
        FileNotFoundError: If ``history`` is a folder without ``metrics.csv``.
    """
    # Read the history, and the metric that picked best/, from the run's output_dir
    if isinstance(history, (str, Path)):
        output_dir = Path(history)
        if not (output_dir / "metrics.csv").exists():
            raise FileNotFoundError(
                f"{output_dir} holds no metrics.csv. Pass the output_dir of a ReIDModel.train run, or the list "
                "train returns."
            )
        history = read_metrics_csv(output_dir)
        if checkpoint_metric == "auto":
            checkpoint_metric = json.loads((output_dir / "training_config.json").read_text())["checkpoint_metric"]

    # re-association reports val/mrr and retrieval reports val/map
    elif checkpoint_metric == "auto":
        if "val/mrr" in history[-1]:
            checkpoint_metric = "mrr"
        elif "val/map" in history[-1]:
            checkpoint_metric = "map"
        else:
            checkpoint_metric = None

    # The columns each panel plots, top to bottom, read from the last row, which holds every column
    columns = list(history[-1])
    curves_by_panel = {"losses": [name for name in columns if name.startswith("train/")]}
    if any(name.startswith("val/") for name in columns):
        curves_by_panel["validation"] = [name for name in columns if name.startswith("val/")]
    curves_by_panel["learning rate"] = ["lr"]

    # The epoch train saved to best/, the highest score after training, a tie going to the later epoch
    best_epoch = None
    if checkpoint_metric is not None:
        trained = [row for row in history if row["epoch"] > 0]
        best_epoch = max(trained, key=lambda row: (row[f"val/{checkpoint_metric}"], row["epoch"]))["epoch"]

    # One panel per group, sharing the epoch axis
    figure, panels = plt.subplots(
        len(curves_by_panel), 1, sharex=True, figsize=(8, 2.5 * len(curves_by_panel)), layout="constrained"
    )
    for axes, (title, names) in zip(panels, curves_by_panel.items()):
        for name in names:
            rows = [row for row in history if name in row]
            axes.plot([row["epoch"] for row in rows], [row[name] for row in rows], marker="o", markersize=3, label=name)
        if best_epoch is not None and title != "learning rate":
            axes.axvline(best_epoch, color="0.4", linestyle="--", linewidth=1)
        if best_epoch is not None and title == "validation":
            # Near the bottom, by the line, since the curves peak at the top there
            axes.text(best_epoch, 0.03, "best epoch ", transform=axes.get_xaxis_transform(), ha="right", color="0.4")
        axes.set_title(title)
        axes.grid(alpha=0.3)
        if len(names) > 1:
            axes.legend()
    panels[-1].set_yscale("log")  # the learning rate

    # Ticks only on whole epochs that ran, since matplotlib can place one at 2.5 or past the last epoch
    panels[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    panels[-1].set_xlim(history[0]["epoch"] - 0.5, history[-1]["epoch"] + 0.5)
    panels[-1].set_xlabel("epoch")
    panels[-1].xaxis.set_major_formatter(_epoch_tick_label)
    return figure


def _epoch_tick_label(epoch: float, position: int) -> str:
    """Label the tick of ``epoch`` on the x axis, reading "before training" for epoch 0."""
    return "before training" if epoch == 0 else f"{epoch:g}"


def select_embeddings_plot_crops(val_data: ReIDSplit, num_people: int) -> ReIDSplit:
    """Pick the validation crops the embeddings plots show.

    They are the ``num_people`` people with the most crops, leaving out ``val_data.gallery_junk_pids``, and up to 20
    crops of each, spread evenly over that person's crops in order. A tie in the number of crops goes to the lower
    person ID.
    """
    # Count the crops of each person, leaving out the junk person IDs
    kept = ~np.isin(val_data.pids, list(val_data.gallery_junk_pids))
    people, crop_counts = np.unique(val_data.pids[kept], return_counts=True)

    # The num_people people with the most crops, and up to 20 crops of each
    most_crops_first = np.argsort(-crop_counts, kind="stable")
    chosen: list[int] = []
    for person in people[most_crops_first[:num_people]]:
        crops = np.flatnonzero(val_data.pids == person)
        chosen.extend(crops[np.linspace(0, len(crops) - 1, min(20, len(crops))).astype(int)])
    return ReIDSplit(
        image_paths=[val_data.image_paths[index] for index in chosen],
        pids=val_data.pids[chosen],
        camids=val_data.camids[chosen],
    )


def save_embeddings_plot(embeddings: np.ndarray, pids: np.ndarray, distance: str, title: str, path: Path) -> None:
    """Save a scatter plot of ``embeddings`` projected onto their 2 principal components, one color per person.

    With the cosine distance, the embeddings are scaled to unit length first.

    Args:
        embeddings: One embedding per crop, shape ``(N, D)``.
        pids: Person ID of each crop, shape ``(N,)``.
        distance: ``"cosine"`` or ``"euclidean"``, the distance validation compares the embeddings with.
        title: Title of the plot, such as ``"before training"`` or ``"epoch 3"``.
        path: PNG file to write. Its folder is created if missing.
    """
    if distance == "cosine":
        embeddings = _l2_normalize(embeddings)
    points = _project_with_pca(embeddings)

    # One color per person, from matplotlib's 10 or 20 distinct colors, or spread along turbo for more people
    people = np.unique(pids)
    if len(people) <= 10:
        colors = colormaps["tab10"].colors
    elif len(people) <= 20:
        colors = colormaps["tab20"].colors
    else:
        colors = colormaps["turbo"](np.linspace(0, 1, len(people)))

    figure = Figure(figsize=(6, 5), layout="constrained")
    axes = figure.subplots()
    for person, color in zip(people, colors):
        person_points = points[pids == person]
        axes.scatter(person_points[:, 0], person_points[:, 1], s=16, color=color, label=str(person))
    axes.set_title(title)
    axes.set_xlabel("principal component 1")
    axes.set_ylabel("principal component 2")
    axes.legend(title="person ID", loc="center left", bbox_to_anchor=(1, 0.5))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)


def show_in_notebook(path: Path) -> None:
    """Show the image at ``path`` in the cell's output when running in a notebook, and do nothing elsewhere."""
    # A notebook runs its code in an IPython kernel, which a script never loads
    if "ipykernel" in sys.modules:
        from IPython.display import Image, display

        display(Image(filename=str(path)))


def _project_with_pca(embeddings: np.ndarray) -> np.ndarray:
    """Project ``embeddings`` onto their 2 principal components, and return the ``(N, 2)`` coordinates."""
    centred = embeddings - embeddings.mean(axis=0)
    _, _, components = np.linalg.svd(centred, full_matrices=False)
    components = components[:2]

    # Make the largest entry of each component positive, as scikit-learn's svd_flip does, so a component and its
    # negative, which fit equally well, always give the same picture
    largest_entries = components[np.arange(2), np.abs(components).argmax(axis=1)]
    components = components * np.sign(largest_entries)[:, np.newaxis]
    return centred @ components.T
