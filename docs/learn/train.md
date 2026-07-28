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

For a runnable MOT17 walkthrough (download, train, retrieval, tracking A/B), see [`notebooks/train_reid_mot17.ipynb`](../../notebooks/train_reid_mot17.ipynb).

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

Each sequence folder needs `gt/gt.txt` and `img1/*.jpg` (optional `seqinfo.ini`):

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
)
print(train_stats.num_identities, train_stats.num_crops)
```

Validation crops (optional, for per-epoch val loss):

```python
val_stats = generate_mot_patches(
    "/path/to/mot17/val",
    "/path/to/crops/val",
    split="full",
)
```

If you already have identity folders from another source, skip this step and pass that root to `train_reid` (or load it with `load_identity_crops`).

`val_data` matches crops by identity folder name. Trackers' `mot17/val` holds later frames of the same sequences, so names like `MOT17-02_5` line up with train. If train and val share no folder names, training raises.

Optional filters: `min_visibility` drops occluded boxes; `min_side` drops tiny crops. Defaults keep all GT rows that pass class and confidence checks.

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
model = ReIDModel.from_pretrained("/path/to/checkpoint")
```

| Field | Role | MOT17 starting point |
| :---- | :--- | :------------------- |
| `pretrained` | Warm-start alias or local checkpoint | `osnet_x1_0_msmt17_combineall` |
| `p`, `k` | PK batch: identities × crops | `8`, `4` (use `4`, `4` for FastReID 384×128) |
| `epochs` | Training length | 30 to 40 for OSNet |
| `freeze_backbone_epochs` | Train head only first | 3 to 5 when warm-starting |
| `identities_per_epoch` | Identity slots per epoch | `"all"` (default) |
| `crops_per_identity` | Crop draws per identity slot | `k` when omitted |
| `num_workers` | DataLoader workers | `0` in notebooks / Colab |

**Epoch budget.** Batches per epoch are `(identities_per_epoch * crops_per_identity) // (p * k)`. Defaults (`"all"` identities, `crops_per_identity=k`) give about `num_identities // p` batches (about 25 for ~200 MOT17 identities with `p=8`). Raise `crops_per_identity` to deepen an epoch without visiting every frame. Lower `identities_per_epoch` for smoke tests.

With `val_data` set, `output_dir` stores the best-epoch checkpoint (`weights.safetensors` and `reid_config.json`). `result.history` has per-epoch train (and val) losses: plot `total` / `val_total` and confirm they trend down before trusting the run.

---

## 4. Validate on gallery retrieval

After training, score embeddings on held-out crops. This is separate from `val_data` loss during training.

```python
from reid import ReIDEvaluator, ReIDModel
from reid.data import build_identity_crop_retrieval_split, load_identity_crops

split = load_identity_crops("/path/to/crops/val")
query, gallery = build_identity_crop_retrieval_split(split, queries_per_id=1)

baseline = ReIDModel.from_pretrained()
finetuned = ReIDModel.from_pretrained("/path/to/checkpoint")

for label, model in [("baseline", baseline), ("finetuned", finetuned)]:
    m = ReIDEvaluator(model).evaluate(query, gallery).metrics
    print(label, f"mAP={m.map:.1f}% Rank-1={m.rank1:.1f}%")
```

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

MOT17 val-half eval in the study uses appearance threshold `0.2` (Table 8). Tracking HOTA / IDF1 on detector crops is the main domain-adaptation signal; gallery mAP alone does not predict ID switches.

The MOT17 notebook compares BoT-SORT with baseline ReID vs fine-tuned ReID on the seven val sequences.
