# ------------------------------------------------------------------------
# re-ID
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""``ReIDModel.train``: the recipe, freezing, the state a run leaves, the output folder, and input guards."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.optim.lr_scheduler import OneCycleLR, StepLR
from torchvision.transforms import CenterCrop

from reid import ReIDModel, ReIDPreprocessing, ReIDSplit, load_identity_folders
from reid.architectures import build_architecture
from reid.architectures.fastreid_sbs import FASTREID_SBS_ARCHITECTURE

from .crop_images import write_identity_crops, write_mot_crop


def _osnet() -> ReIDModel:
    """OSNet x0.25 on 64x32 crops, small enough to train in a test on CPU."""
    return ReIDModel("osnet_x0_25", preprocessing=ReIDPreprocessing(input_size=(64, 32)), device="cpu")


def _small_backbone() -> nn.Sequential:
    """One convolution with an 8-number embedding and no ``trainable_when_frozen``."""
    return nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.AdaptiveAvgPool2d(1), nn.Flatten())


def _small_module() -> ReIDModel:
    """A model built from ``_small_backbone``, so it has no architecture name."""
    return ReIDModel(_small_backbone(), preprocessing=ReIDPreprocessing(input_size=(64, 32)), device="cpu")


@pytest.fixture
def crops(tmp_path: Path) -> ReIDSplit:
    write_identity_crops(tmp_path / "crops", num_ids=4, num_per=4)
    return load_identity_folders(tmp_path / "crops")


def _missing_crops(pids: list[int]) -> ReIDSplit:
    """A split whose images don't exist, so a check that runs after loading fails differently."""
    return ReIDSplit(
        image_paths=[f"missing/{index}.jpg" for index in range(len(pids))],
        pids=np.array(pids),
        camids=np.zeros(len(pids), dtype=int),
    )


def _train(model: ReIDModel, data: ReIDSplit, **settings: object) -> list[dict[str, float]]:
    """Train one batch of 2 identities x 2 crops with no warmup or freezing, unless ``settings`` say otherwise."""
    defaults = {
        "epochs": 1,
        "steps_per_epoch": 1,
        "warmup_epochs": 0,
        "freeze_backbone_epochs": 0,
        "identities_per_batch": 2,
        "crops_per_identity": 2,
        "verbose": False,
    }
    return model.train(data, **{**defaults, **settings})


Run = tuple[ReIDModel, list[dict[str, float]], Path]


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    """One 2-epoch run on 4 identities with the default batches per epoch, written to an output folder."""
    root = tmp_path_factory.mktemp("run")
    write_identity_crops(root / "crops", num_ids=4, num_per=4)
    model = _osnet()
    history = _train(
        model, load_identity_folders(root / "crops"), epochs=2, steps_per_epoch=None, output_dir=root / "out"
    )
    return model, history, root / "out"


class _ConstantLoss(nn.Module):
    """A loss that is always ``value``, to tell which loss ``train`` used."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return outputs.sum() * 0 + self.value  # tied to the outputs, so train can call backward() on it


class TestTraining:
    """The training recipe, and what a run does to the model."""

    def test_a_failed_run_restores_eval_mode_and_requires_grad(self) -> None:
        model = _osnet()

        with pytest.raises(FileNotFoundError, match="missing/"):
            _train(model, _missing_crops([0, 0, 1, 1]), epochs=2, freeze_backbone_epochs=1)  # fails in frozen epoch 1

        assert not model._backbone.training
        assert all(parameter.requires_grad for parameter in model._backbone.parameters())

    def test_learns_colour_coded_identities(self, tmp_path: Path) -> None:
        """The last epoch's loss falls below 0.8 of the first when training works."""
        rng = np.random.default_rng(0)
        for identity, colour in enumerate([(255, 40, 40), (40, 255, 40), (40, 40, 255), (255, 255, 40)]):
            (tmp_path / f"id_{identity}").mkdir()
            for index in range(8):
                crop = np.full((48, 24, 3), colour) + rng.normal(0, 20, (48, 24, 3))
                Image.fromarray(np.clip(crop, 0, 255).astype("uint8")).save(
                    tmp_path / f"id_{identity}" / f"{index}.jpg"
                )

        history = _train(
            _osnet(),
            load_identity_folders(tmp_path),
            epochs=2,
            steps_per_epoch=8,
            lr=1e-3,
            identities_per_batch=4,
            crops_per_identity=4,
        )

        assert history[-1]["train/loss"] < 0.8 * history[0]["train/loss"]

    @pytest.mark.parametrize(
        "epochs, warmup_epochs, expected_lr",
        [
            pytest.param(8, 2, [1 / 2, 1, 1, 1, 1, 1, 0.1, 0.01], id="warmup-then-drops-after-epochs-6-and-7"),
            pytest.param(1, 0, [1], id="one-epoch-run-trains-at-lr"),
            pytest.param(3, 0, [1, 1, 0.1], id="the-second-drop-comes-an-epoch-after-the-first"),
            pytest.param(
                8,
                7,
                [1 / 7, 2 / 7, 3 / 7, 4 / 7, 5 / 7, 6 / 7, 1, 0.1],
                id="a-drop-inside-the-warmup-moves-to-its-end",
            ),
        ],
    )
    def test_lr_warms_up_then_drops_after_three_quarters_and_seven_eighths(
        self, crops: ReIDSplit, epochs: int, warmup_epochs: int, expected_lr: list[float]
    ) -> None:
        history = _train(_small_module(), crops, epochs=epochs, warmup_epochs=warmup_epochs, lr=1.0)

        assert [row["lr"] for row in history] == pytest.approx(expected_lr)

    @pytest.mark.parametrize(
        "warmup_epochs, expected_lr",
        [
            pytest.param(2, [0.5, 1, 1, 0.5, 0.25], id="after-a-2-epoch-warmup"),
            pytest.param(0, [1, 0.5, 0.25, 0.125, 0.0625], id="from-the-first-epoch-without-warmup"),
        ],
    )
    def test_lr_scheduler_takes_over_after_the_warmup(
        self, crops: ReIDSplit, warmup_epochs: int, expected_lr: list[float]
    ) -> None:
        history = _train(
            _small_module(),
            crops,
            epochs=5,
            warmup_epochs=warmup_epochs,
            lr=1.0,
            lr_scheduler=StepLR,
            lr_scheduler_kwargs={"step_size": 1, "gamma": 0.5},
        )

        # StepLR starts at lr in the first epoch after the warmup, then halves it after every epoch
        assert [row["lr"] for row in history] == pytest.approx(expected_lr)

    def test_warns_when_the_warmup_is_longer_than_the_run(self, crops: ReIDSplit) -> None:
        with pytest.warns(UserWarning, match="the learning rate never reaches lr"):
            _train(_small_module(), crops, epochs=2, warmup_epochs=3)

    def test_leaves_out_a_sequence_with_a_single_identity_with_a_warning(self, tmp_path: Path) -> None:
        paths = [str(write_mot_crop(tmp_path, "a", track_id, frame)) for track_id in (1, 2) for frame in range(2)]
        crops = ReIDSplit(
            image_paths=[*paths, "missing/b_3/b_000000.jpg", "missing/b_3/b_000001.jpg"],  # b's crops aren't on disk
            pids=np.array([0, 0, 1, 1, 2, 2]),  # identity 2 alone in sequence b
            camids=np.zeros(6, dtype=int),
        )

        with pytest.warns(UserWarning, match="Sequences with a single identity will be left out of the batches.*: b."):
            history = _train(_small_module(), crops, steps_per_epoch=10, sample_per_sequence=True)

        # b's crops don't exist, so any batch that drew sequence b would have raised FileNotFoundError
        assert len(history) == 1

    def test_trains_with_the_optimizer_passed(self, crops: ReIDSplit) -> None:
        built_with = []

        class RecordingSGD(torch.optim.SGD):
            def __init__(self, params: object, **kwargs: object) -> None:
                super().__init__(params, **kwargs)
                built_with.append(kwargs)

        _train(
            _small_module(), crops, lr=0.1, weight_decay=0.0, optimizer=RecordingSGD, optimizer_kwargs={"momentum": 0.9}
        )

        assert built_with == [{"lr": 0.1, "weight_decay": 0.0, "momentum": 0.9}]

    def test_same_seed_repeats_the_run(self, crops: ReIDSplit) -> None:
        first = _small_module()
        second = copy.deepcopy(first)

        _train(first, crops, seed=7)
        _train(second, crops, seed=7)

        for name, value in first._backbone.state_dict().items():
            assert torch.equal(value, second._backbone.state_dict()[name]), name

    def test_total_loss_is_the_weighted_sum_of_the_losses_passed(self, crops: ReIDSplit) -> None:
        history = _train(
            _small_module(),
            crops,
            classification_loss=_ConstantLoss(7.0),
            embedding_loss=_ConstantLoss(5.0),
            classification_loss_weight=2.0,
            embedding_loss_weight=3.0,
        )

        assert history[0]["train/loss"] == pytest.approx(29.0)  # 2 x 7 + 3 x 5
        assert history[0]["train/classification_loss"] == pytest.approx(7.0)  # each part is logged before its weight
        assert history[0]["train/embedding_loss"] == pytest.approx(5.0)

    def test_prints_one_line_per_epoch(self, crops: ReIDSplit, capsys: pytest.CaptureFixture[str]) -> None:
        _train(_small_module(), crops, epochs=2, lr=1e-3, verbose=True)

        lines = capsys.readouterr().out.splitlines()

        # a 2-epoch run drops lr to a tenth after epoch 1
        assert [line.split("  ")[:2] for line in lines] == [["epoch 1/2", "lr 1.0e-03"], ["epoch 2/2", "lr 1.0e-04"]]


class TestFreezing:
    """Which layers train: the ``trainable_when_frozen`` layers in the first epochs, every layer after them."""

    def test_frozen_epochs_keep_the_batchnorm_statistics_of_frozen_layers(self, crops: ReIDSplit) -> None:
        model = _osnet()

        _train(model, crops, epochs=2, freeze_backbone_epochs=1)

        assert model._backbone.conv1.bn.num_batches_tracked.item() == 1  # a frozen layer, so it updates in epoch 2 only
        assert model._backbone.fc[1].num_batches_tracked.item() == 2  # OSNet's fc keeps training while frozen

    @pytest.mark.filterwarnings("ignore:freeze_backbone_epochs=1 is at least epochs=1")
    @pytest.mark.parametrize(
        "epochs, conv1_trains",
        [pytest.param(1, False, id="not-in-the-frozen-epoch"), pytest.param(2, True, id="in-the-epoch-after-it")],
    )
    def test_a_frozen_layer_trains_only_after_the_frozen_epochs(
        self, crops: ReIDSplit, epochs: int, conv1_trains: bool
    ) -> None:
        model = _osnet()
        conv1_before = [parameter.detach().clone() for parameter in model._backbone.conv1.parameters()]

        _train(model, crops, epochs=epochs, freeze_backbone_epochs=1)  # conv1 is frozen in epoch 1

        conv1_after = list(model._backbone.conv1.parameters())
        conv1_changed = any(not torch.equal(before, after) for before, after in zip(conv1_before, conv1_after))
        assert conv1_changed == conv1_trains

    @pytest.mark.parametrize("architecture", ["osnet_x0_25", FASTREID_SBS_ARCHITECTURE])
    def test_trainable_when_frozen_names_layers_of_the_architecture(self, architecture: str) -> None:
        backbone = build_architecture(architecture)

        layer_names = {name for name, _ in backbone.named_children()}

        assert set(backbone.trainable_when_frozen) <= layer_names  # a stale name would make train raise

    def test_warns_when_every_epoch_is_frozen(self, crops: ReIDSplit) -> None:
        with pytest.warns(UserWarning, match="trains only the classification layer"):
            _train(_osnet(), crops, epochs=1, freeze_backbone_epochs=1)


class TestOutputDir:
    """``output_dir`` holds ``last/``, ``metrics.csv`` and ``training_config.json``."""

    def test_last_reloads_as_the_trained_model(self, run: Run) -> None:
        model, _, output_dir = run

        reloaded = ReIDModel.from_pretrained(str(output_dir / "last"), device="cpu")

        for name, value in model._backbone.state_dict().items():
            assert torch.equal(value, reloaded._backbone.state_dict()[name]), name

    def test_metrics_csv_holds_the_history(self, run: Run) -> None:
        _, history, output_dir = run

        with (output_dir / "metrics.csv").open() as file:
            rows = list(csv.DictReader(file))

        assert [{name: float(value) for name, value in row.items()} for row in rows] == history

    def test_training_config_records_the_run(self, run: Run) -> None:
        _, _, output_dir = run

        config = json.loads((output_dir / "training_config.json").read_text())

        assert config == {
            "architecture": "osnet_x0_25",
            "epochs": 2,
            "steps_per_epoch": 4,  # one pass over 16 crops in batches of 2 identities x 2 crops
            "identities_per_batch": 2,
            "crops_per_identity": 2,
            "sample_per_sequence": False,
            "lr": 3e-4,
            "warmup_epochs": 0,
            "weight_decay": 5e-4,
            "optimizer": "Adam",
            "optimizer_kwargs": None,
            "lr_scheduler": None,
            "lr_scheduler_kwargs": None,
            "freeze_backbone_epochs": 0,
            "classification_loss": "CrossEntropyLoss()",
            "label_smoothing": 0.1,
            "embedding_loss": "TripletLoss(margin=0.3)",
            "classification_loss_weight": 1.0,
            "embedding_loss_weight": 1.0,
            "augmentations": [
                "RandomHorizontalFlip(p=0.5)",
                "ColorJitter(brightness=(0.8, 1.2), contrast=(0.85, 1.15), saturation=(0.85, 1.15), hue=None)",
                "RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=random, inplace=False)",
            ],
            "num_workers": 0,
            "mixed_precision": False,  # mixed precision runs only on CUDA, and this run is on the CPU
            "seed": 42,
            "num_identities": 4,
            "num_crops": 16,
        }

    @pytest.mark.parametrize("earlier_file", ["last/weights.safetensors", "metrics.csv"])
    def test_rejects_an_output_dir_holding_another_run(
        self, crops: ReIDSplit, tmp_path: Path, earlier_file: str
    ) -> None:
        earlier = tmp_path / "out" / earlier_file
        earlier.parent.mkdir(parents=True)
        earlier.write_text("earlier run")

        with pytest.raises(ValueError, match="already holds"):
            _train(_osnet(), crops, output_dir=tmp_path / "out")

        files = [
            path.relative_to(tmp_path / "out").as_posix() for path in (tmp_path / "out").rglob("*") if path.is_file()
        ]
        assert files == [earlier_file]
        assert earlier.read_text() == "earlier run"

    def test_last_of_a_model_built_from_a_module_reloads_with_that_module(
        self, crops: ReIDSplit, tmp_path: Path
    ) -> None:
        model = _small_module()
        _train(model, crops, output_dir=tmp_path / "out")

        # last/ records the module's class name, so loading it back takes a module of that class
        reloaded = ReIDModel.from_pretrained(
            str(tmp_path / "out" / "last"), architecture=_small_backbone(), device="cpu"
        )

        assert json.loads((tmp_path / "out" / "last" / "reid_config.json").read_text())["architecture"] == "Sequential"
        for name, value in model._backbone.state_dict().items():
            assert torch.equal(value, reloaded._backbone.state_dict()[name]), name

    def test_last_of_a_model_built_from_a_module_asks_for_that_module(self, crops: ReIDSplit, tmp_path: Path) -> None:
        _train(_small_module(), crops, output_dir=tmp_path / "out")

        with pytest.raises(ValueError, match="built from your own module Sequential"):
            ReIDModel.from_pretrained(str(tmp_path / "out" / "last"), device="cpu")


class TestInputGuards:
    """Arguments that can't work raise before any crop is read."""

    def test_rejects_a_folder_path(self) -> None:
        with pytest.raises(TypeError, match="load_identity_folders"):
            _train(_osnet(), "crops/train")

    def test_rejects_fewer_identities_than_identities_per_batch(self) -> None:
        with pytest.raises(ValueError, match="identities_per_batch=4 needs at least 4 identities.*got 3"):
            _train(_osnet(), _missing_crops([0, 0, 1, 1, 2, 2]), identities_per_batch=4)

    @pytest.mark.parametrize("argument", ["identities_per_batch", "crops_per_identity"])
    def test_batches_need_two_identities_and_two_crops_each(self, argument: str) -> None:
        with pytest.raises(ValueError, match=f"{argument} must be an integer of at least 2"):
            _train(_osnet(), _missing_crops([0, 0, 1, 1]), **{argument: 1})

    def test_rejects_a_scheduler_train_cannot_step(self) -> None:
        with pytest.raises(ValueError, match="lr_scheduler=OneCycleLR can't run in train"):
            _train(_osnet(), _missing_crops([0, 0, 1, 1]), lr_scheduler=OneCycleLR)

    def test_rejects_lr_scheduler_kwargs_without_a_scheduler(self) -> None:
        with pytest.raises(ValueError, match="lr_scheduler_kwargs needs an lr_scheduler"):
            _train(_osnet(), _missing_crops([0, 0, 1, 1]), lr_scheduler_kwargs={"T_max": 3})

    def test_rejects_trainable_when_frozen_naming_a_missing_module(self) -> None:
        model = _small_module()
        model._backbone.trainable_when_frozen = ("fc",)  # an nn.Sequential's children are named "0", "1", ...

        with pytest.raises(ValueError, match=r"trainable_when_frozen names \['fc'\]"):
            _train(model, _missing_crops([0, 0, 1, 1]))

    def test_rejects_augmentations_that_change_the_crop_size(self) -> None:
        with pytest.raises(ValueError, match="augmentations give 32x32 crops, but the model's input size is 64x32"):
            _train(_osnet(), _missing_crops([0, 0, 1, 1]), augmentations=[CenterCrop(32)])

    def test_rejects_crops_not_named_by_sequence_with_sample_per_sequence(self) -> None:
        with pytest.raises(ValueError, match="sample_per_sequence=True reads each crop's sequence from its path"):
            _train(_osnet(), _missing_crops([0, 0, 1, 1]), sample_per_sequence=True)  # crops named missing/0.jpg, ...

    def test_rejects_sample_per_sequence_when_no_sequence_has_two_identities(self) -> None:
        crops = ReIDSplit(
            image_paths=["missing/a_1/a_000000.jpg", "missing/a_1/a_000001.jpg", "missing/b_2/b_000000.jpg"],
            pids=np.array([0, 0, 1]),  # identity 0 alone in sequence a, identity 1 alone in sequence b
            camids=np.zeros(3, dtype=int),
        )

        with pytest.raises(ValueError, match="needs a sequence with at least 2 identities"):
            _train(_osnet(), crops, sample_per_sequence=True)
