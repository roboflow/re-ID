# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Smoke tests for ``train_reid`` and training-related backbone freeze."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import supervision as sv
import torch
from PIL import Image
from safetensors import safe_open

from reid import ReIDModel
from reid.architectures import build_architecture
from reid.preprocessing import ReIDPreprocessing
from reid.training import TrainableBackbone, TrainConfig, train_reid


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

    def test_crops_per_identity_follows_k_after_replace(self) -> None:
        base = TrainConfig(p=8, k=4)
        swept = dataclasses.replace(base, k=8)

        assert base.crops_per_identity is None
        assert base.num_batches(320) == 40
        assert swept.num_batches(320) == 40
        assert dataclasses.replace(base, crops_per_identity=8).num_batches(320) == 80


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

        with safe_open(out_dir / "weights.safetensors", framework="pt") as saved:
            saved_keys = set(saved.keys())
        assert not any(key.startswith("classifier.") for key in saved_keys)
        assert saved_keys == {key for key in result.model._backbone.state_dict() if not key.startswith("classifier.")}

        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        detections = sv.Detections(xyxy=np.array([[10, 10, 40, 80]], dtype=float))
        embeddings = result.model.extract_features(detections, frame)
        assert embeddings.shape == (1, 512)

        reloaded = ReIDModel.from_pretrained(str(out_dir)).extract_features(detections, frame)
        np.testing.assert_allclose(reloaded, embeddings, atol=1e-5)

    def test_history_lr_is_the_trained_lr(self, tmp_path: Path) -> None:
        """history['lr'] reports the LR the epoch trained at, not the post-step value."""
        data_root = tmp_path / "crops"
        _make_crops(data_root, num_ids=4, num_per=4)
        config = TrainConfig(
            epochs=2, p=2, k=2, lr=0.1, warmup_epochs=2, num_workers=0, amp=False, device="cpu", log_interval=0
        )

        result = train_reid(data_root, config, architecture="osnet_x0_25", pretrained=None)

        assert result.history[0]["lr"] == pytest.approx(0.05)
        assert result.history[1]["lr"] == pytest.approx(0.1)

    def test_timm_backbone_trains_and_round_trips(self, tmp_path: Path) -> None:
        """timm trunks train behind TimmEncoder and reload into the plain timm model."""
        data_root = tmp_path / "crops"
        _make_crops(data_root, num_ids=4, num_per=4)
        raw = build_architecture("timm:resnet18", num_classes=0)
        checkpoint = tmp_path / "resnet18.pth"
        torch.save(raw.state_dict(), checkpoint)
        config = TrainConfig(
            epochs=1, p=2, k=2, warmup_epochs=0, num_workers=0, amp=False, device="cpu", log_interval=0
        )
        out_dir = tmp_path / "checkpoint"

        result = train_reid(
            data_root,
            config,
            architecture="timm:resnet18",
            pretrained=str(checkpoint),
            output_dir=out_dir,
        )
        reloaded = ReIDModel.from_pretrained(str(out_dir))

        frame = np.zeros((96, 64, 3), dtype=np.uint8)
        detections = sv.Detections(xyxy=np.array([[8, 8, 40, 88]], dtype=float))
        np.testing.assert_allclose(
            reloaded.extract_features(detections, frame),
            result.model.extract_features(detections, frame),
            atol=1e-5,
        )

    def test_freeze_backbone_epochs_keeps_trunk_fixed(self, tmp_path: Path) -> None:
        """With every epoch frozen, warm-started trunk weights must not move while the head trains."""
        data_root = tmp_path / "crops"
        _make_crops(data_root, num_ids=4, num_per=4)
        reference = build_architecture("osnet_x0_25", num_classes=0)
        checkpoint = tmp_path / "reference.pth"
        torch.save(reference.state_dict(), checkpoint)
        config = TrainConfig(
            epochs=1,
            p=2,
            k=2,
            warmup_epochs=0,
            num_workers=0,
            amp=False,
            device="cpu",
            log_interval=0,
            freeze_backbone_epochs=1,
        )

        result = train_reid(data_root, config, architecture="osnet_x0_25", pretrained=str(checkpoint))

        trained = result.model._backbone.state_dict()
        source = reference.state_dict()
        trunk = [key for key in trained if key in source and not key.startswith("fc.")]
        assert trunk
        assert all(torch.equal(trained[key], source[key]) for key in trunk)
        assert any(not torch.equal(trained[key], source[key]) for key in trained if key.startswith("fc."))

    def test_learns_separable_toy_identities(self, tmp_path: Path) -> None:
        """Two epochs on colour-coded identities must cut the loss well below its start.

        Loss ratio is the discriminator: ~0.67 when the loop trains, ~0.98 with
        updates disabled. Retrieval is deliberately not asserted — a random
        backbone already retrieves solid colours perfectly.
        """
        rng = np.random.default_rng(0)
        data_root = tmp_path / "crops"
        for i, color in enumerate([(255, 40, 40), (40, 255, 40), (40, 40, 255), (255, 255, 40)]):
            identity_dir = data_root / f"id_{i}"
            identity_dir.mkdir(parents=True)
            for j in range(8):
                crop = np.full((64, 32, 3), color, dtype=np.float32) + rng.normal(0, 20, (64, 32, 3))
                Image.fromarray(np.clip(crop, 0, 255).astype("uint8")).save(identity_dir / f"{j}.jpg")

        config = TrainConfig(
            epochs=2,
            p=4,
            k=4,
            warmup_epochs=0,
            num_workers=0,
            amp=False,
            device="cpu",
            log_interval=0,
            identities_per_epoch=16,
        )
        result = train_reid(data_root, config, architecture="osnet_x0_25", pretrained=None)

        assert result.history[-1]["total"] < 0.85 * result.history[0]["total"]

    def test_warm_start_rejects_mismatched_checkpoint(self, tmp_path: Path) -> None:
        """A checkpoint matching ~0% of the architecture must raise, not train from random init."""
        data_root = tmp_path / "crops"
        _make_crops(data_root, num_ids=4, num_per=4)
        wrong_ckpt = tmp_path / "wrong.pth"
        torch.save({"backbone.stem.weight": torch.randn(4, 3, 3, 3)}, wrong_ckpt)
        config = TrainConfig(
            epochs=1, p=2, k=2, warmup_epochs=0, num_workers=0, amp=False, device="cpu", log_interval=0
        )

        with pytest.raises(ValueError, match="matched only"):
            train_reid(data_root, config, architecture="osnet_x0_25", pretrained=str(wrong_ckpt))

    def test_preprocessing_round_trips_through_save(self, tmp_path: Path) -> None:
        data_root = tmp_path / "crops"
        _make_crops(data_root, num_ids=4, num_per=4)
        config = TrainConfig(
            epochs=1, p=2, k=2, warmup_epochs=0, num_workers=0, amp=False, device="cpu", log_interval=0
        )
        preprocessing = ReIDPreprocessing(resize_mode="letterbox", interpolation="bicubic")
        out_dir = tmp_path / "checkpoint"

        train_reid(
            data_root,
            config,
            architecture="osnet_x0_25",
            pretrained=None,
            preprocessing=preprocessing,
            output_dir=out_dir,
        )
        reloaded = ReIDModel.from_pretrained(str(out_dir))

        assert reloaded._preprocessing.resize_mode == "letterbox"
        assert reloaded._preprocessing.interpolation == preprocessing.interpolation

    def test_center_loss_is_reported_and_optimized(self, tmp_path: Path) -> None:
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
            log_interval=0,
            use_center=True,
        )
        result = train_reid(data_root, config, architecture="osnet_x0_25", pretrained=None)

        assert "center" in result.history[0]

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
        val_totals = [epoch["val_total"] for epoch in result.history]
        assert result.best_epoch == int(np.argmin(val_totals))
        assert result.best_val_total == pytest.approx(min(val_totals))
        assert (out_dir / "weights.safetensors").exists()
