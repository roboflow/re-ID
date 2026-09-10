---
title: Fine-tune on tracking crops — train_reid workflow | reid
description: Build per-identity crop datasets from MOT-format sequences, fine-tune with train_reid, validate with gallery metrics, and load the checkpoint into BoT-SORT.
---

# Fine-tune on tracking crops

Fine-tune a ReID encoder on per-identity person crops, then reload the checkpoint for gallery evaluation or BoT-SORT.

`train_reid` consumes an identity-folder crop tree (`<root>/<identity>/*.jpg`) or a `ReIDSplit`. One built-in way to create those folders is `generate_mot_patches`, which reads any MOT-layout sequence root (`gt/gt.txt` + `img1/`). That covers MOT17, SportsMOT, and custom MOT-format data. The examples below use MOT17; swap paths for other datasets.

**What you'll learn:**

- Prepare MOT-format sequences and build crop folders
- Configure and run `train_reid`
- Check training loss and gallery mAP / Rank-1
- Load fine-tuned weights into BoT-SORT

Runnable notebooks: [`train_reid_mot17.ipynb`](https://github.com/roboflow/re-ID/blob/main/notebooks/train_reid_mot17.ipynb) (MOT17 train / val tracking A/B) and [`train_reid_dancetrack.ipynb`](https://github.com/roboflow/re-ID/blob/main/notebooks/train_reid_dancetrack.ipynb) (DanceTrack train, val metrics, Codabench test).

---

## Install

```bash
pip install reid
```

```python
from reid.data.mot import generate_mot_patches
from reid.training import TrainConfig, train_reid
```

---

## 1. Get MOT-format sequences

Each sequence folder needs `gt/gt.txt`, `img1/*.jpg`, and `seqinfo.ini` (required by the `train_half` / `val_half` splits to place the frame cutoff):

```text
mot17/train/
├── MOT17-02-FRCNN/
│   ├── gt/gt.txt
│   ├── img1/000001.jpg
│   └── seqinfo.ini
└── ...
```

Download a supported benchmark with Trackers (`mot17` or `sportsmot`):

```bash
pip install trackers
trackers download mot17 --split train,val --asset annotations,frames -o /path/to/data
```

Trackers MOT17 packages are already temporal halves (`mot17/train/` and `mot17/val/`). When you crop them in the next step, use `split="full"` so frames are not halved again. SportsMOT downloads follow the same layout under `sportsmot/`.

If you have a single MOT-format root with the full sequence in `gt.txt`, use `split="train_half"` for training crops and `split="val_half"` for validation crops from that same root.

---

## 2. Build crop datasets

`generate_mot_patches` writes one folder per identity (`<sequence>_<track_id>`) with one `.jpg` crop per GT box:

```text
crops/train/
├── MOT17-02_1/
│   ├── MOT17-02_000001.jpg
│   └── ...
└── MOT17-02_3/
    └── ...
```

Train crops:

```python
from reid.data.mot import generate_mot_patches

train_stats = generate_mot_patches(
    "/path/to/mot17/train",
    "/path/to/crops/train",
    split="full",
    class_ids=[1],
)
print(train_stats.num_identities, train_stats.num_crops)
```

Validation crops (optional, for per-epoch val loss):

```python
val_stats = generate_mot_patches(
    "/path/to/mot17/val",
    "/path/to/crops/val",
    split="full",
    class_ids=[1],
)
```

If you already have identity folders from another source, skip this step and point `ReIDDataset` at that root.

`val_data` matches crops by identity folder name. Trackers' `mot17/val` holds later frames of the same sequences, so names like `MOT17-02_5` line up with train. If train and val share no folder names, training raises.

Optional filters: `class_ids` keeps only the listed ground-truth classes; `min_visibility` drops occluded boxes; `min_side` drops tiny crops. Apart from the confidence flag, the defaults keep every GT row.

Class ids are dataset-specific, so `generate_mot_patches` assumes no taxonomy: `class_ids=None` (the default) crops every class. MOT17 needs `class_ids=[1]`, since it reserves other ids for vehicles, static persons, distractors, and occluders — training on those would treat furniture and reflections as identities. DanceTrack and SportsMOT label every box `1`, so the default is already correct there.

---

## 3. Train

```python
import logging

from reid import ReIDModel
from reid.training import TrainConfig, train_reid

logging.basicConfig(level=logging.INFO)

config = TrainConfig(
    epochs=30,
    p=8,
    k=4,
    lr=3e-4,
    warmup_epochs=5,
    lr_milestones=(20, 27),
    freeze_backbone_epochs=3,
    num_workers=0,
    log_interval=10,
)

result = train_reid(
    "/path/to/crops/train",
    config,
    pretrained="osnet_x1_0_msmt17_combineall",
    output_dir="/path/to/checkpoint",
    val_data="/path/to/crops/val",
)

print(result.best_epoch, result.best_val_total)
```

Or drive the whole lifecycle from the model, passing one crop set:

```python
from reid import ReIDModel
from reid.data import ReIDDataset

crops = ReIDDataset("/path/to/crops/train")
val = ReIDDataset("/path/to/crops/val")

model = ReIDModel.from_pretrained("osnet_x1_0_msmt17_combineall", device="cuda")
result = model.train(crops, val_split=val, epochs=30, p=8, k=4, output_dir="/path/to/checkpoint")
metrics = model.evaluate(val, queries_per_id=1)
model = ReIDModel.from_pretrained("/path/to/checkpoint/best")
```

| Field                    | Role                                 | MOT17 starting point                         |
| :----------------------- | :----------------------------------- | :------------------------------------------- |
| `pretrained`             | Warm-start alias or local checkpoint | `osnet_x1_0_msmt17_combineall`               |
| `p`, `k`                 | PK batch: identities × crops         | `8`, `4` (use `4`, `4` for FastReID 384×128) |
| `epochs`                 | Training length                      | 30 to 40 for OSNet                           |
| `freeze_backbone_epochs` | Train head only first                | 3 to 5 when warm-starting                    |
| `identities_per_epoch`   | Identity slots per epoch             | `"all"` (default)                            |
| `crops_per_identity`     | Crop draws per identity slot         | `k` when omitted                             |
| `num_workers`            | DataLoader workers                   | `0` in notebooks / Colab                     |
| `use_center`             | Add the center-loss term             | `False` (see below)                          |

**Epoch budget.** Batches per epoch are `(identities_per_epoch * crops_per_identity) // (p * k)`. Defaults (`"all"` identities, `crops_per_identity=k`) give about `num_identities // p` batches (about 25 for ~200 MOT17 identities with `p=8`). Raise `crops_per_identity` to deepen an epoch without visiting every frame. Lower `identities_per_epoch` for smoke tests.

`output_dir` holds two checkpoints, each with `weights.safetensors` and `reid_config.json`: `last/` is rewritten every epoch, and with `val_data` set `best/` is rewritten whenever `val_total` improves. Both are written during training, so an interrupted run still leaves usable weights. Note that `best` is the lowest validation loss over identities that also appear in training, which is not the same as the encoder that generalises best to unseen identities, so compare the two before shipping one. `result.history` has per-epoch train (and val) losses: plot `total` / `val_total` and confirm they trend down before trusting the run.

**Center loss (opt-in).** `train_reid` optimizes cross-entropy plus a batch-hard triplet term. `use_center=True` adds the third Bag-of-Tricks term, which pulls embeddings toward a learnable per-identity center. It is off by default because batch-hard triplet already enforces intra-class compactness, so the two terms largely overlap.

Two differences from the Bag-of-Tricks reference are worth knowing before you compare runs. Centers here share the main Adam optimizer rather than getting their own SGD optimizer, and Adam normalizes gradient magnitude — so `center_weight` changes how hard the term pulls the embeddings but does not change how fast the centers themselves move. Only `center_lr` does that, and it needs to stay large (the `0.5` default): centers start from `randn` and must travel a long way onto the embedding manifold, so at `0.01` they barely leave their initialisation within a typical run.

---

## 4. Validate on gallery retrieval

After training, score embeddings on held-out crops. This is separate from `val_data` loss during training.

```python
from reid import ReIDModel
from reid.data import ReIDDataset

crops = ReIDDataset("/path/to/crops/val")
model = ReIDModel.from_pretrained("/path/to/checkpoint")
result = model.evaluate(crops, queries_per_id=1)
print(result.metrics)
```

`evaluate` derives the query/gallery partition from identity folders. Benchmark datasets ship their own splits, so pass those explicitly instead: `model.evaluate(*load_market1501(root))`.

On the MOT17 notebook protocol (train on `mot17/train` crops, retrieve on `mot17/val` crops), scenes and IDs can overlap across time. Treat mAP / Rank-1 as a temporal diagnostic, not an open-set ReID score like Market-1501. For identity-disjoint holdout, partition with `build_identity_crop_holdout_split` before training. Metric definitions: [ReID Metrics](metrics.md).

---

## 5. Use in tracking

```bash
pip install 'trackers[reid]'
```

```python
from reid import ReIDModel
from trackers import BoTSORTTracker

reid_model = ReIDModel.from_pretrained("/path/to/checkpoint")
tracker = BoTSORTTracker(reid_model=reid_model, appearance_threshold=0.2)
```

The MOT17 notebook fixes `appearance_threshold=0.2` for the val-half A/B (the BoT-SORT paper selects 0.25 on the same split; Aharon et al., 2022, [arXiv:2206.14651](https://arxiv.org/abs/2206.14651)). Tracking HOTA / IDF1 on detector crops is the main domain-adaptation signal; gallery mAP alone does not predict ID switches.

The MOT17 notebook compares BoT-SORT with baseline ReID vs fine-tuned ReID on the seven val sequences.

---

## 6. Fine-tune results

Numbers below come from the DanceTrack Colab notebook: OSNet warm-started from MSMT17, trained on DanceTrack train-half GT crops, best epoch by val loss on train val-half, then BoT-SORT on YOLOX dets with `appearance_threshold=0.25`.

### DanceTrack test (Codabench)

| Tracker                                                                                                                        | HOTA | IDF1 | MOTA |
| :----------------------------------------------------------------------------------------------------------------------------- | :--: | :--: | :--: |
| BoT-SORT default (trackers DanceTrack Codabench baseline)                                                                      | 57.8 | 57.9 | 92.2 |
| BoT-SORT + OSNet DanceTrack FT ([notebook](https://github.com/roboflow/re-ID/blob/main/notebooks/train_reid_dancetrack.ipynb)) | 58.5 | 58.9 | 92.1 |

Delta vs default: +0.7 HOTA, +1.0 IDF1, MOTA -0.1.
