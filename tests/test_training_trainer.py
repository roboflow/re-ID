# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Smoke tests for ``train_reid`` and training-related backbone freeze."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import supervision as sv
import torch
from PIL import Image

from reid.architectures import build_architecture
from reid.training import TrainConfig, train_reid
from reid.training.trainer import TrainableBackbone


def _make_crops(root: Path, num_ids: int, num_per: int) -> None:
    rng = np.random.default_rng(0)
    for i in range(num_ids):
        identity_dir = root / f"id_{i:02d}"
        identity_dir.mkdir(parents=True)
        for j in range(num_per):
            array = (rng.random((64, 32, 3)) * 255).astype("uint8")
            Image.fromarray(array).save(identity_dir / f"{j}.jpg")


class TestTrainConfig:
    """Tests for :class:`TrainConfig` validation."""

    def test_rejects_degenerate_pk_with_triplet(self) -> None:
        with pytest.raises(ValueError, match="p >= 2 and k >= 2"):
            TrainConfig(p=1, k=4, triplet_weight=1.0)


class TestTrainableBackbone:
    """Tests for the backbone contract :func:`train_reid` requires."""

    def test_shipped_architecture_satisfies_contract(self) -> None:
        backbone = build_architecture("osnet_x0_25", num_classes=4)
        assert isinstance(backbone, TrainableBackbone)

    def test_plain_module_does_not_satisfy_contract(self) -> None:
        assert not isinstance(torch.nn.Linear(4, 4), TrainableBackbone)


class TestBackboneFreeze:
    """Tests for architecture ``set_backbone_frozen`` used by training."""

    def test_frozen_trunk_keeps_batchnorm_stats_fixed(self) -> None:
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


class TestTrainReid:
    """End-to-end smoke for :func:`train_reid`."""

    def test_saves_and_embeds(self, tmp_path: Path) -> None:
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
        assert result.best_epoch is None
        assert (out_dir / "weights.safetensors").exists()
        assert (out_dir / "reid_config.json").exists()

        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        detections = sv.Detections(xyxy=np.array([[10, 10, 40, 80]], dtype=float))
        embeddings = result.model.extract_features(detections, frame)
        assert embeddings.shape == (1, 512)

    def test_val_keeps_best_epoch(self, tmp_path: Path) -> None:
        train_root = tmp_path / "train"
        val_root = tmp_path / "val"
        _make_crops(train_root, num_ids=4, num_per=4)
        _make_crops(val_root, num_ids=4, num_per=4)

        config = TrainConfig(
            epochs=2,
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
            train_root,
            config,
            architecture="osnet_x0_25",
            pretrained=None,
            output_dir=out_dir,
            val_data=val_root,
        )

        assert len(result.history) == 2
        assert "val_total" in result.history[0]
        assert "val_ce" in result.history[0]
        assert "val_triplet" in result.history[0]
        assert result.best_epoch is not None
        assert result.best_val_total is not None
        assert result.best_epoch in {0, 1}
        assert (out_dir / "weights.safetensors").exists()
