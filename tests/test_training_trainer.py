# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Smoke tests for ``train_reid``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import supervision as sv
import torch
from PIL import Image

from reid.architectures import build_architecture
from reid.training import TrainConfig, train_reid


def _make_crops(root: Path, num_ids: int, num_per: int) -> None:
    rng = np.random.default_rng(0)
    for i in range(num_ids):
        identity_dir = root / f"id_{i:02d}"
        identity_dir.mkdir(parents=True)
        for j in range(num_per):
            array = (rng.random((64, 32, 3)) * 255).astype("uint8")
            Image.fromarray(array).save(identity_dir / f"{j}.jpg")


def test_train_config_rejects_degenerate_pk_with_triplet() -> None:
    with pytest.raises(ValueError, match="p >= 2 and k >= 2"):
        TrainConfig(p=1, k=4, triplet_weight=1.0)


def test_frozen_trunk_keeps_batchnorm_stats_fixed() -> None:
    """Freeze must stop trunk BatchNorm running-stat updates, not only grads."""
    backbone = build_architecture("osnet_x0_25", num_classes=4)
    backbone.train()
    backbone.set_backbone_frozen(True)

    bn = backbone.conv1.bn
    before = bn.running_mean.clone()
    backbone.forward_embeddings_logits(torch.randn(4, 3, 256, 128))
    assert torch.equal(before, bn.running_mean)

    backbone.set_backbone_frozen(False)
    backbone.forward_embeddings_logits(torch.randn(4, 3, 256, 128))
    assert not torch.equal(before, bn.running_mean)


def test_train_reid_smoke_saves_and_embeds(tmp_path: Path) -> None:
    """One epoch of random-init OSNet writes a loadable checkpoint and embeds."""
    data_root = tmp_path / "crops"
    _make_crops(data_root, num_ids=4, num_per=4)

    config = TrainConfig(
        epochs=1,
        p=2,
        k=2,
        warmup_epochs=0,
        num_workers=0,
        amp=False,
        device="cpu",
        log_interval=1,
    )
    out_dir = tmp_path / "checkpoint"
    result = train_reid(
        data_root,
        config,
        architecture="osnet_x0_25",
        pretrained=None,
        output_dir=out_dir,
    )

    assert result.num_classes == 4
    assert len(result.history) == 1
    assert "total" in result.history[0]
    assert (out_dir / "weights.safetensors").exists()
    assert (out_dir / "reid_config.json").exists()

    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    detections = sv.Detections(xyxy=np.array([[10, 10, 40, 80]], dtype=float))
    embeddings = result.model.extract_features(detections, frame)
    assert embeddings.shape == (1, 512)
