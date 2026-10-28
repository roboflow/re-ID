# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""ReID model / architecture / loader tests.

Requires the ``reid`` package (torch, timm, huggingface_hub, safetensors).
Installed and run in CI via ``uv sync --all-extras`` and ``pytest tests/ -m 'not slow'``.
Curated weight downloads live in the ``@pytest.mark.slow`` test below.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import supervision as sv
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file, save_file
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

import reid.catalog
from reid import ReIDModel
from reid.architectures import build_architecture, default_preprocessing_for_architecture
from reid.architectures.fastreid_sbs import FASTREID_SBS_ARCHITECTURE, FastReIDSBSResNeSt50
from reid.catalog import (
    DEFAULT_MODEL,
    FASTREID_MOT17_SBS50,
    list_aliases,
    resolve_model_card,
)
from reid.loaders import load_state_dict, resolve_weights
from reid.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    LETTERBOX_PAD_VALUE,
    Interpolation,
    ReIDPreprocessing,
)


def _save_checkpoint(state_dict: object, suffix: str = ".pth") -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        tmp_path = f.name
    torch.save(state_dict, tmp_path)
    return tmp_path


def _mock_downloads(monkeypatch: pytest.MonkeyPatch, hosted: Path, *, cache: Path) -> list[str]:
    """Mock the download ``reid`` uses, so tests need no network.

    A request for ``https://<host>/<path>`` copies ``hosted/<host>/<path>``, and one for
    ``https://<host>/<path>?<query>`` copies ``hosted/<host>/<query>/<path>``, or answers 404
    when that file is missing. The weights cache moves to ``cache``, so each test starts
    empty. Returns the URLs ``reid`` requests, in order.
    """
    requested: list[str] = []

    def download(url: str, dst: str, *args: object, **kwargs: object) -> None:
        requested.append(url)
        parsed = urllib.parse.urlparse(url)
        served = hosted / parsed.netloc / parsed.query / parsed.path.lstrip("/")
        if not served.is_file():
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        shutil.copyfile(served, dst)

    monkeypatch.setenv("REID_CACHE_DIR", str(cache))
    monkeypatch.setattr(torch.hub, "download_url_to_file", download)
    return requested


def _tiny_encoder():
    class _TinyEncoder(nn.Module):
        def forward(self, x):
            return x.flatten(1)[:, :16]

    return _TinyEncoder()


def _assert_full_load_logged(caplog: pytest.LogCaptureFixture) -> None:
    assert "0 missing, 0 unused" in caplog.text


class TestReIDPreprocessing:
    """Preprocessing geometry, color conversion, and config serialization."""

    def test_crop_transform_matches_torchreid(self) -> None:
        """OSNet was trained and benchmarked with torchreid's test transform."""
        crop = Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 140, 3), dtype=np.uint8))
        torchreid_transform = Compose([Resize((256, 128)), ToTensor(), Normalize(IMAGENET_MEAN, IMAGENET_STD)])

        assert torch.equal(ReIDPreprocessing().build_crop_transform()(crop), torchreid_transform(crop))

    def test_stretch_resize_matches_target_size(self) -> None:
        crop = Image.fromarray(np.zeros((100, 50, 3), dtype=np.uint8))
        out = np.asarray(ReIDPreprocessing(input_size=(384, 128), resize_mode="stretch").resize_image(crop))
        assert out.shape == (384, 128, 3)

    def test_letterbox_preserves_aspect_and_pads(self) -> None:
        crop = Image.fromarray(np.full((200, 100, 3), 255, dtype=np.uint8))
        out = np.asarray(ReIDPreprocessing(input_size=(384, 128), resize_mode="letterbox").resize_image(crop))
        assert out.shape == (384, 128, 3)
        assert out[0, 0, 0] == 255
        assert out[-1, 0, 0] == LETTERBOX_PAD_VALUE

    def test_unknown_interpolation_raises(self) -> None:
        with pytest.raises(ValueError, match="lanczos"):
            ReIDPreprocessing(interpolation="lanczos")

    def test_unknown_resize_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="resize_mode"):
            ReIDPreprocessing(resize_mode="crop")  # type: ignore[arg-type]

    def test_prepare_crop_array_swaps_bgr_frames_only(self) -> None:
        bgr = np.array([[[255, 0, 0]]], dtype=np.uint8)  # blue in BGR
        p = ReIDPreprocessing(to_rgb=True)
        rgb = p.prepare_crop_array(bgr, from_bgr_frame=True)
        assert rgb[0, 0, 0] == 0
        assert rgb[0, 0, 2] == 255
        unchanged = p.prepare_crop_array(bgr, from_bgr_frame=False)
        np.testing.assert_array_equal(unchanged, bgr)

    def test_every_field_survives_the_config_roundtrip(self) -> None:
        """Every field differs from its default, so a field dropped from ``to_dict`` fails the comparison."""
        preprocessing = ReIDPreprocessing(
            input_size=(128, 64),
            mean=(0.1, 0.2, 0.3),
            std=(0.4, 0.5, 0.6),
            interpolation="bicubic",
            to_rgb=False,
            resize_mode="letterbox",
            pad_value=7,
        )

        restored = ReIDPreprocessing.from_dict(preprocessing.to_dict())

        assert restored == preprocessing
        assert restored.interpolation is Interpolation.BICUBIC

    def test_from_dict_ignores_legacy_normalize_embeddings(self) -> None:
        """Old reid_config.json files carrying normalize_embeddings still load."""
        legacy = ReIDPreprocessing(input_size=(128, 64), to_rgb=False).to_dict()
        legacy["normalize_embeddings"] = True  # removed field from an older version
        p = ReIDPreprocessing.from_dict(legacy)
        assert p == ReIDPreprocessing(input_size=(128, 64), to_rgb=False)
        assert not hasattr(p, "normalize_embeddings")


class TestRegistry:
    """Architecture defaults, model aliases, and load-recipe resolution."""

    def test_list_aliases(self) -> None:
        aliases = list_aliases()
        assert aliases == sorted(
            [
                DEFAULT_MODEL,
                FASTREID_MOT17_SBS50,
                "osnet_x1_0_mot17",
                "osnet_x1_0_sportsmot",
                "osnet_x1_0_dancetrack",
                "osnet_x1_0_soccernet",
            ]
        )

    def test_resolve_default_alias(self) -> None:
        resolved = resolve_model_card(DEFAULT_MODEL)
        assert resolved is not None
        card, config_preprocessing = resolved
        assert card.architecture == "osnet_x1_0"
        assert card.weights is not None
        assert card.domain_warning is not None
        assert config_preprocessing is None

    def test_resolve_fastreid_mot17_alias(self) -> None:
        resolved = resolve_model_card(FASTREID_MOT17_SBS50)
        assert resolved is not None
        card, config_preprocessing = resolved
        assert card.architecture == FASTREID_SBS_ARCHITECTURE
        assert card.weights is not None
        assert card.domain_warning is not None
        assert config_preprocessing is None

    @pytest.mark.parametrize(
        "architecture, input_size",
        [("osnet_x1_0", (256, 128)), (FASTREID_SBS_ARCHITECTURE, (384, 128)), ("timm:resnet50", (256, 128))],
    )
    def test_architecture_default_preprocessing(self, architecture: str, input_size: tuple[int, int]) -> None:
        """Every architecture stretches crops to its own input size; ``timm:`` backbones use the OSNet default."""
        expected = ReIDPreprocessing(input_size=input_size, resize_mode="stretch", to_rgb=True)
        assert default_preprocessing_for_architecture(architecture) == expected

    def test_local_dir_with_config(self, tmp_path) -> None:
        config = {
            "architecture": "osnet_x1_0",
            "preprocessing": ReIDPreprocessing().to_dict(),
        }
        (tmp_path / "reid_config.json").write_text(json.dumps(config))
        resolved = resolve_model_card(str(tmp_path))
        assert resolved is not None
        card, config_preprocessing = resolved
        assert card.architecture == "osnet_x1_0"
        assert config_preprocessing is not None
        assert config_preprocessing.input_size == (256, 128)

    def test_hf_repo_with_config(self, tmp_path, monkeypatch) -> None:
        config = {"architecture": "osnet_x1_0", "preprocessing": ReIDPreprocessing(input_size=(128, 64)).to_dict()}
        (tmp_path / "reid_config.json").write_text(json.dumps(config))
        requests = []

        def download(repo_id: str, filename: str, revision: str | None = None) -> str:
            requests.append((repo_id, filename, revision))
            return str(tmp_path / filename)

        monkeypatch.setattr(reid.catalog, "hf_hub_download", download)

        card, config_preprocessing = resolve_model_card("hf://org/repo@v1")

        assert requests == [("org/repo", "reid_config.json", "v1")]
        assert card.architecture == "osnet_x1_0"
        assert card.weights == "hf://org/repo@v1/weights.safetensors"
        assert config_preprocessing.input_size == (128, 64)

    def test_local_dir_without_config_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="reid_config.json"):
            resolve_model_card(str(tmp_path))

    def test_bare_path_without_architecture_raises(self) -> None:
        """from_pretrained on a bare weights file without architecture → ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            tmp_path = f.name
        try:
            # Write a minimal torch save so the file exists
            torch.save({}, tmp_path)
            with pytest.raises(ValueError, match="architecture"):
                ReIDModel.from_pretrained(tmp_path)
        finally:
            os.unlink(tmp_path)


class TestLoaders:
    """Checkpoint path resolution, remapping, and state-dict loading."""

    def test_rejects_weights_from_another_architecture(self, tmp_path) -> None:
        weights = tmp_path / "weights.pth"
        torch.save(nn.Linear(4, 2).state_dict(), weights)  # no parameter OSNet has

        with pytest.raises(ValueError, match=r"matched only 0 of 565 'osnet_x0_25' weight tensors \(required >= 50%\)"):
            ReIDModel.from_pretrained(str(weights), architecture="osnet_x0_25", device="cpu")

    def test_saved_model_must_load_completely(self, tmp_path) -> None:
        ReIDModel("osnet_x0_25", device="cpu").save(str(tmp_path))
        weights = load_file(tmp_path / "weights.safetensors")
        save_file({k: v for k, v in list(weights.items())[1:]}, tmp_path / "weights.safetensors")  # drop one tensor

        with pytest.raises(ValueError, match=r"matched only 564 of 565 .* \(required >= 100%\)"):
            ReIDModel.from_pretrained(str(tmp_path), device="cpu")

    def test_saved_model_needs_its_weights_file(self, tmp_path) -> None:
        ReIDModel("osnet_x0_25", device="cpu").save(str(tmp_path))
        (tmp_path / "weights.safetensors").unlink()

        with pytest.raises(FileNotFoundError, match="Weights file not found"):
            ReIDModel.from_pretrained(str(tmp_path), device="cpu")

    def test_saved_fastreid_model_reloads_with_the_same_weights(self, tmp_path) -> None:
        model = ReIDModel(FASTREID_SBS_ARCHITECTURE, device="cpu")
        model.save(str(tmp_path))

        reloaded = ReIDModel.from_pretrained(str(tmp_path), device="cpu")

        original, restored = model._backbone.state_dict(), reloaded._backbone.state_dict()
        assert all(torch.equal(original[key], restored[key]) for key in original)

    def test_missing_local_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="Weights file not found"):
            resolve_weights("/nonexistent/path/to/weights.pth")

    def test_malformed_hf_url_raises(self) -> None:
        with pytest.raises(ValueError, match="hf://"):
            resolve_weights("hf://only_one_part")

    def test_url_weights_download_once_then_load_from_cache(self, tmp_path, monkeypatch) -> None:
        ReIDModel("osnet_x0_25", device="cpu").save(str(tmp_path / "hosted" / "storage.example.com" / "reid"))
        downloads = _mock_downloads(monkeypatch, tmp_path / "hosted", cache=tmp_path / "cache")
        url = "https://storage.example.com/reid/weights.safetensors"

        for _ in range(2):
            ReIDModel.from_pretrained(url, architecture="osnet_x0_25", device="cpu")

        assert downloads == [url]
        assert (tmp_path / "cache" / "weights" / "storage.example.com" / "reid" / "weights.safetensors").is_file()

    def test_urls_ending_in_the_same_file_name_are_cached_apart(self, tmp_path, monkeypatch) -> None:
        """Model folders all hold a ``weights.safetensors``; a cache keyed by file name would mix them up."""
        for name in ("mot17", "soccernet"):
            ReIDModel("osnet_x0_25", device="cpu").save(str(tmp_path / "hosted" / "storage.example.com" / name))
        _mock_downloads(monkeypatch, tmp_path / "hosted", cache=tmp_path / "cache")

        mot17 = resolve_weights("https://storage.example.com/mot17/weights.safetensors")
        soccernet = resolve_weights("https://storage.example.com/soccernet/weights.safetensors")

        assert mot17 != soccernet
        assert open(mot17, "rb").read() != open(soccernet, "rb").read()

    def test_model_folder_url_loads_its_config_and_weights(self, tmp_path, monkeypatch) -> None:
        """A ``ReIDModel.save`` folder served over https loads like a local one: its config sets the model."""
        saved = ReIDModel("osnet_x0_25", preprocessing=ReIDPreprocessing(input_size=(128, 64)), device="cpu")
        saved.save(str(tmp_path / "hosted" / "storage.example.com" / "models" / "custom"))
        downloads = _mock_downloads(monkeypatch, tmp_path / "hosted", cache=tmp_path / "cache")

        loaded = ReIDModel.from_pretrained("https://storage.example.com/models/custom/", device="cpu")

        assert downloads == [
            "https://storage.example.com/models/custom/reid_config.json",
            "https://storage.example.com/models/custom/weights.safetensors",
        ]
        assert loaded.preprocessing == saved.preprocessing
        original, restored = saved._backbone.state_dict(), loaded._backbone.state_dict()
        assert all(torch.equal(original[key], restored[key]) for key in original)

    def test_model_folder_url_without_a_config_raises(self, tmp_path, monkeypatch) -> None:
        _mock_downloads(monkeypatch, tmp_path / "hosted", cache=tmp_path / "cache")

        with pytest.raises(ValueError, match="no reid_config.json"):
            ReIDModel.from_pretrained("https://storage.example.com/models/empty/", device="cpu")

    def test_failed_url_download_names_the_url(self, tmp_path, monkeypatch) -> None:
        _mock_downloads(monkeypatch, tmp_path / "hosted", cache=tmp_path / "cache")

        with pytest.raises(RuntimeError, match="https://storage.example.com/missing.pth"):
            resolve_weights("https://storage.example.com/missing.pth")

    def test_url_query_is_part_of_the_cache_key(self, tmp_path, monkeypatch) -> None:
        """``?version=1`` and ``?version=2`` are different files and must not share a cache entry."""
        for version in ("version=1", "version=2"):
            ReIDModel("osnet_x0_25", device="cpu").save(
                str(tmp_path / "hosted" / "storage.example.com" / version / "reid")
            )
        downloads = _mock_downloads(monkeypatch, tmp_path / "hosted", cache=tmp_path / "cache")
        urls = [f"https://storage.example.com/reid/weights.safetensors?{v}" for v in ("version=1", "version=2")]

        first, second = (resolve_weights(url) for url in urls)

        assert downloads == urls
        assert first.endswith(".safetensors") and second.endswith(".safetensors")
        assert open(first, "rb").read() != open(second, "rb").read()

    def test_model_folder_url_keeps_its_query_and_drops_its_fragment(self, tmp_path, monkeypatch) -> None:
        ReIDModel("osnet_x0_25", device="cpu").save(
            str(tmp_path / "hosted" / "storage.example.com" / "token=abc" / "models" / "custom")
        )
        downloads = _mock_downloads(monkeypatch, tmp_path / "hosted", cache=tmp_path / "cache")

        ReIDModel.from_pretrained("https://storage.example.com/models/custom/?token=abc#readme", device="cpu")

        assert downloads == [
            "https://storage.example.com/models/custom/reid_config.json?token=abc",
            "https://storage.example.com/models/custom/weights.safetensors?token=abc",
        ]

    def test_failed_url_download_keeps_a_file_another_download_finished(self, tmp_path, monkeypatch) -> None:
        """Two processes can download the same URL; the one that fails must not delete the other's result."""
        monkeypatch.setenv("REID_CACHE_DIR", str(tmp_path / "cache"))
        cached = tmp_path / "cache" / "weights" / "storage.example.com" / "reid" / "weights.safetensors"

        def finished_elsewhere_then_fail(url: str, dst: str, *args: object, **kwargs: object) -> None:
            cached.write_bytes(b"weights from the other download")
            raise urllib.error.HTTPError(url, 503, "Service Unavailable", None, None)

        monkeypatch.setattr(torch.hub, "download_url_to_file", finished_elsewhere_then_fail)

        with pytest.raises(RuntimeError, match="storage.example.com"):
            resolve_weights("https://storage.example.com/reid/weights.safetensors")
        assert cached.read_bytes() == b"weights from the other download"
        assert [p.name for p in cached.parent.iterdir()] == ["weights.safetensors"]

    def test_empty_url_download_is_not_cached(self, tmp_path, monkeypatch) -> None:
        """A server that answers with an empty file once must not break every later load."""
        hosted = tmp_path / "hosted" / "storage.example.com" / "reid" / "weights.safetensors"
        hosted.parent.mkdir(parents=True)
        hosted.touch()
        downloads = _mock_downloads(monkeypatch, tmp_path / "hosted", cache=tmp_path / "cache")
        url = "https://storage.example.com/reid/weights.safetensors"

        with pytest.raises(RuntimeError, match="empty"):
            resolve_weights(url)
        ReIDModel("osnet_x0_25", device="cpu").save(str(hosted.parent))
        ReIDModel.from_pretrained(url, architecture="osnet_x0_25", device="cpu")

        assert downloads == [url, url]

    def test_url_checkpoint_that_needs_pickle_raises(self, tmp_path, monkeypatch) -> None:
        """A downloaded ``.pth`` never reaches ``weights_only=False``; the same file loaded locally still does."""
        checkpoint = tmp_path / "hosted" / "storage.example.com" / "reid" / "osnet.pth"
        checkpoint.parent.mkdir(parents=True)
        backbone = ReIDModel("osnet_x0_25", device="cpu")._backbone
        torch.save({"state_dict": backbone.state_dict(), "args": argparse.Namespace(lr=0.1)}, checkpoint)
        _mock_downloads(monkeypatch, tmp_path / "hosted", cache=tmp_path / "cache")

        with pytest.raises(ValueError, match="safetensors"):
            ReIDModel.from_pretrained("https://storage.example.com/reid/osnet.pth", architecture="osnet_x0_25")
        with pytest.warns(UserWarning, match="weights_only=False"):
            ReIDModel.from_pretrained(str(checkpoint), architecture="osnet_x0_25", device="cpu")

    def test_load_state_dict_mismatch_warns(self) -> None:
        source = nn.Linear(4, 2)
        target = nn.Linear(8, 4)
        tmp_path = _save_checkpoint(source.state_dict())
        try:
            with pytest.warns(UserWarning, match="weights likely do not match"):
                load_state_dict(target, tmp_path, torch.device("cpu"))
        finally:
            os.unlink(tmp_path)

    def test_partial_load_in_high_band_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """A 50–99% match (missing keys, not a full mismatch) still warns."""

        class _TwoLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = nn.Linear(4, 2)
                self.b = nn.Linear(4, 2)

        target = _TwoLayer()
        # Checkpoint provides 3 of the 4 params (75%): b.bias is missing.
        partial = {
            "a.weight": target.a.weight.detach().clone(),
            "a.bias": target.a.bias.detach().clone(),
            "b.weight": target.b.weight.detach().clone(),
        }
        tmp_path = _save_checkpoint(partial)
        try:
            with caplog.at_level(logging.INFO, logger="reid.loaders"):
                with pytest.warns(UserWarning, match="Partial load"):
                    load_state_dict(target, tmp_path, torch.device("cpu"))
            assert "b.bias" in caplog.text
            assert "missing=" in caplog.text
        finally:
            os.unlink(tmp_path)

    def test_generic_loader_copies_backbone_prefixed_weights(self, caplog: pytest.LogCaptureFixture) -> None:
        """A ``backbone.`` prefix is a real module name, not a wrapper to strip, so the weights must arrive."""

        class _BackboneModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = nn.Linear(4, 2)

        source, target = _BackboneModel(), _BackboneModel()
        tmp_path = _save_checkpoint(source.state_dict())
        try:
            with caplog.at_level(logging.INFO, logger="reid.loaders"):
                load_state_dict(target, tmp_path, torch.device("cpu"))
            _assert_full_load_logged(caplog)
            for key, value in source.state_dict().items():
                torch.testing.assert_close(target.state_dict()[key], value)
        finally:
            os.unlink(tmp_path)

    def test_osnet_backbone_loads_with_full_match(self, tmp_path, caplog: pytest.LogCaptureFixture) -> None:
        """Inference OSNet has no classifier, so strict checkpoint loads succeed."""
        source = build_architecture("osnet_x0_25")
        assert not any(key.startswith("classifier.") for key in source.state_dict())

        path = tmp_path / "osnet.pth"
        torch.save(source.state_dict(), path)

        target = build_architecture("osnet_x0_25")
        with caplog.at_level(logging.INFO, logger="reid.loaders"):
            load_state_dict(
                target,
                str(path),
                torch.device("cpu"),
                architecture="osnet_x0_25",
                required_match_fraction=1.0,
            )
        _assert_full_load_logged(caplog)

    def test_fastreid_checkpoint_loads_fully_and_skips_the_classifier(self) -> None:
        """A BoT-SORT checkpoint (``module.`` prefix, ``heads.*`` keys, identity classifier) loads every weight.

        GeM ``p`` must come from ``heads.pool_layer.p``, not the 3.0 constructor default, and the classifier
        (``heads.weight``) has no place in the inference model.
        """
        mot17_gem_p = 1.7194522619247437
        model = cast(FastReIDSBSResNeSt50, build_architecture(FASTREID_SBS_ARCHITECTURE))
        assert model.pool.p.item() == pytest.approx(3.0)

        checkpoint: dict = {"heads.pool_layer.p": torch.tensor([mot17_gem_p]), "heads.weight": torch.zeros(487, 2048)}
        for key, value in model.state_dict().items():
            if key.startswith("backbone."):
                checkpoint[f"module.{key}"] = value
            elif key.startswith("bottleneck."):
                checkpoint[f"heads.bottleneck.0.{key[len('bottleneck.') :]}"] = value

        tmp_path = _save_checkpoint(checkpoint)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                load_state_dict(
                    model,
                    tmp_path,
                    torch.device("cpu"),
                    architecture=FASTREID_SBS_ARCHITECTURE,
                    required_match_fraction=1.0,
                )
            assert model.pool.p.item() == pytest.approx(mot17_gem_p)
        finally:
            os.unlink(tmp_path)

    def test_fastreid_sbs_last_stride_patch(self) -> None:
        """``layer4[0]`` must match FastReID LAST_STRIDE=1 (not raw timm output_stride=16)."""
        model = cast(FastReIDSBSResNeSt50, build_architecture(FASTREID_SBS_ARCHITECTURE))
        block = cast(Any, cast(Any, model.backbone).layer4[0])
        assert isinstance(block.downsample[0], nn.AvgPool2d)
        assert block.downsample[0].kernel_size == (1, 1) or block.downsample[0].kernel_size == 1
        assert block.downsample[0].stride == (1, 1) or block.downsample[0].stride == 1
        assert isinstance(block.avd_last, nn.AvgPool2d)
        assert block.avd_last.kernel_size == (3, 3) or block.avd_last.kernel_size == 3
        assert block.avd_last.stride == (1, 1) or block.avd_last.stride == 1

    def test_fastreid_sbs_forward_shape(self) -> None:
        model = build_architecture(FASTREID_SBS_ARCHITECTURE)
        model.eval()
        with torch.inference_mode():
            out = model(torch.randn(2, 3, 384, 128))
        assert out.shape == (2, 2048)


class TestModelSmoke:
    """Core ``ReIDModel`` construction and extraction smoke tests."""

    def test_out_of_frame_box_embeds_its_in_frame_part(self) -> None:
        """A box sticking out of the frame is clipped to the part inside it, not replaced by a blank crop."""
        model = ReIDModel(_tiny_encoder(), device="cpu")
        frame = np.random.default_rng(0).integers(0, 255, size=(128, 128, 3), dtype=np.uint8)

        outside = model.extract_embeddings(sv.Detections(xyxy=np.array([[-40.0, -40.0, 10.0, 10.0]])), frame)
        inside = model.extract_embeddings(sv.Detections(xyxy=np.array([[0.0, 0.0, 10.0, 10.0]])), frame)

        np.testing.assert_array_equal(outside, inside)

    def test_empty_detections_returns_empty(self) -> None:
        model = ReIDModel(_tiny_encoder(), device="cpu")
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        embs = model.extract_embeddings(sv.Detections.empty(), frame)
        assert embs.shape == (0, 0)

    def test_empty_paths_returns_empty(self) -> None:
        model = ReIDModel(_tiny_encoder(), device="cpu")
        embs = model.extract_embeddings_from_paths([])
        assert embs.shape == (0, 0)

    def test_from_pretrained_without_source_points_to_the_constructor(self) -> None:
        with pytest.raises(ValueError, match=r"use ReIDModel\('osnet_x0_25'\)"):
            ReIDModel.from_pretrained(architecture="osnet_x0_25", device="cpu")

    def test_constructor_builds_without_downloads(self) -> None:
        """The constructor builds an OSNet model with random weights and no downloads."""
        # osnet_x0_25 is a clean-room architecture with no external weights, so
        # this builds entirely offline (random init).
        model = ReIDModel("osnet_x0_25", device="cpu")
        frame = np.zeros((256, 128, 3), dtype=np.uint8)
        dets = sv.Detections(xyxy=np.array([[0, 0, 128, 256]], dtype=np.float32))
        embs = model.extract_embeddings(dets, frame)
        assert embs.ndim == 2
        assert embs.shape[0] == 1
        assert embs.dtype == np.float32
        norms = np.linalg.norm(embs, axis=1)
        assert not np.allclose(norms, np.ones_like(norms), atol=1e-5)


class TestExtractContract:
    """Extraction parity, batching, eval mode, and raw-embedding behavior."""

    def _tiny_model(self, preprocessing):
        return ReIDModel(_tiny_encoder(), preprocessing=preprocessing, device="cpu")

    def test_init_forces_backbone_eval(self) -> None:
        """A backbone left in train() mode is switched to eval() on construction."""

        class _WithBN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bn = nn.BatchNorm1d(16)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.bn(x.flatten(1)[:, :16])

        backbone = _WithBN()
        backbone.train()
        assert backbone.training is True
        model = ReIDModel(backbone, device="cpu")
        assert model._backbone.training is False

    def test_extract_embeddings_sub_batches(self) -> None:
        """extract_embeddings splits crops into batch_size chunks for the forward pass."""

        class _Counting(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.batch_sizes: list[int] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.batch_sizes.append(int(x.shape[0]))
                return x.flatten(1)[:, :16]

        backbone = _Counting()
        model = ReIDModel(backbone, device="cpu")
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        boxes = np.array([[0, 0, 32, 32]] * 5, dtype=np.float32)
        dets = sv.Detections(xyxy=boxes)

        embs = model.extract_embeddings(dets, frame, batch_size=2)
        assert embs.shape[0] == 5
        assert backbone.batch_sizes == [2, 2, 1]

    def test_extract_embeddings_rejects_bad_batch_size(self) -> None:
        model = self._tiny_model(ReIDPreprocessing())
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        dets = sv.Detections(xyxy=np.array([[0, 0, 32, 32]], dtype=np.float32))
        with pytest.raises(ValueError, match="batch_size"):
            model.extract_embeddings(dets, frame, batch_size=0)

    def test_frame_and_path_extract_parity(self, tmp_path) -> None:
        """Same crop via a frame box vs. an on-disk file yields identical embeddings."""
        # to_rgb=False keeps frame bytes as-is; PIL path loads RGB on disk the same way.
        preprocessing = ReIDPreprocessing(input_size=(256, 128), to_rgb=False)
        model = self._tiny_model(preprocessing)

        rng = np.random.default_rng(0)
        crop = rng.integers(0, 255, size=(256, 128, 3), dtype=np.uint8)
        crop_path = tmp_path / "crop.png"
        Image.fromarray(crop).save(crop_path)  # PNG is lossless

        dets = sv.Detections(xyxy=np.array([[0, 0, 128, 256]], dtype=np.float32))
        emb_frame = model.extract_embeddings(dets, crop)
        emb_path = model.extract_embeddings_from_paths([str(crop_path)])

        assert emb_frame.shape == emb_path.shape
        np.testing.assert_allclose(emb_frame, emb_path, atol=1e-6)

    def test_extract_returns_raw_embeddings(self, tmp_path) -> None:
        """extract_* always returns raw embeddings; normalisation is a distance-time concern."""
        model = self._tiny_model(ReIDPreprocessing(to_rgb=False))

        rng = np.random.default_rng(1)
        crop = rng.integers(0, 255, size=(256, 128, 3), dtype=np.uint8)
        crop_path = tmp_path / "crop.png"
        Image.fromarray(crop).save(crop_path)

        path_embs = model.extract_embeddings_from_paths([str(crop_path)])
        assert not np.allclose(np.linalg.norm(path_embs, axis=1), 1.0, atol=1e-3)

        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        dets = sv.Detections(xyxy=np.array([[0, 0, 64, 64]], dtype=np.float32))
        frame_embs = model.extract_embeddings(dets, frame)
        assert not np.allclose(np.linalg.norm(frame_embs, axis=1), 1.0, atol=1e-3)


class TestRoundTrip:
    """``save`` / ``from_pretrained`` round-trip behavior."""

    def test_save_and_reload_produces_identical_embeddings(self, tmp_path) -> None:
        # osnet_x0_25 builds offline; its random init is fixed for this process,
        # so the saved weights reload to byte-identical parameters.
        model_a = ReIDModel("osnet_x0_25", device="cpu")
        save_dir = str(tmp_path / "saved_model")
        model_a.save(save_dir)

        assert os.path.exists(os.path.join(save_dir, "weights.safetensors"))
        assert os.path.exists(os.path.join(save_dir, "reid_config.json"))

        model_b = ReIDModel.from_pretrained(save_dir, device="cpu")
        assert model_b.preprocessing == model_a.preprocessing

        frame = np.zeros((256, 128, 3), dtype=np.uint8)
        dets = sv.Detections(xyxy=np.array([[0, 0, 128, 256]], dtype=np.float32))
        embs_a = model_a.extract_embeddings(dets, frame)
        embs_b = model_b.extract_embeddings(dets, frame)
        np.testing.assert_allclose(embs_a, embs_b, atol=1e-5)


@pytest.mark.slow
def test_default_osnet_alias_loads_with_finite_output() -> None:
    """Download ``DEFAULT_MODEL`` and smoke the default encoder path.

    Curated aliases require a 100% state-dict match; loading must not raise.
    """
    model = ReIDModel.from_pretrained(DEFAULT_MODEL, device="cpu")
    frame = np.zeros((256, 128, 3), dtype=np.uint8)
    dets = sv.Detections(xyxy=np.array([[0.0, 0.0, 128.0, 256.0]], dtype=np.float32))

    embs = model.extract_embeddings(dets, frame)
    assert embs.shape == (1, 512)
    assert np.isfinite(embs).all()
    assert model.preprocessing.input_size == (256, 128)


@pytest.mark.slow
def test_fastreid_mot17_alias_loads_with_finite_output() -> None:
    """Download ``fastreid_mot17_sbs50`` and smoke the BoT-SORT encoder path.

    ``extract_embeddings`` returns raw BNNeck features; cosine similarity is
    applied at distance time in ``reid.distance_matrix``.
    """
    model = ReIDModel.from_pretrained("fastreid_mot17_sbs50", device="cpu")
    frame = np.zeros((384, 128, 3), dtype=np.uint8)
    dets = sv.Detections(xyxy=np.array([[0.0, 0.0, 128.0, 384.0]], dtype=np.float32))

    embs = model.extract_embeddings(dets, frame)
    assert embs.shape == (1, 2048)
    assert np.isfinite(embs).all()
    assert model.preprocessing.input_size == (384, 128)


class TestModelToDevice:
    """``ReIDModel.to`` must move the backbone with the bookkeeping."""

    def test_to_moves_the_backbone_and_returns_self(self) -> None:
        model = ReIDModel(torch.nn.Linear(4, 4), device="cpu")

        # The meta device holds no data, so the move is observable on a CPU-only machine.
        returned = model.to("meta")

        assert returned is model
        assert model.device == torch.device("meta")
        assert all(parameter.device.type == "meta" for parameter in model._backbone.parameters())

    def test_failed_move_keeps_the_recorded_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = ReIDModel(torch.nn.Linear(4, 4), device="cpu")

        def cannot_move(*args: object, **kwargs: object) -> torch.nn.Module:
            raise RuntimeError("cannot move")

        monkeypatch.setattr(model._backbone, "to", cannot_move)

        with pytest.raises(RuntimeError, match="cannot move"):
            model.to("meta")

        assert model.device == torch.device("cpu")
