---
title: Train a Re-ID model for tracking
description: Fine-tune a Re-ID model on crops cut from tracking video, with batches drawn from one sequence, and validate it the way a tracker uses it.
---

# Train a model for tracking

> **Video placeholder.** Header video, no caption: BoT-SORT on a held-out SportsMOT clip, the general model and the fine-tuned model side by side, at the moment the general model swaps two players' IDs and the fine-tuned one keeps them. An mp4 with a poster image, as `reid-gallery-retrieval.mp4`, from the tracking tutorial's run.

A tracker swaps identities when two objects look alike, such as two players of the same team crossing. A Re-ID model fine-tuned on your own footage learns the differences between those objects, so the tracker keeps their identities through crossings and occlusions.

Training a Re-ID model for tracking needs special attention in two places:

- **Turning the tracking dataset into crops.** A tracking dataset has a box for every object in every frame, so it gives far more crops than a Re-ID dataset, and neighboring frames repeat nearly the same image. An epoch over all of them is long, and most of it adds little.
- **Sampling the batches.** In single-camera tracking, track IDs only hold inside their own sequence, and the objects a tracker confuses are the ones that share a scene, so each batch should come from one sequence.

Everything else, from the pretrained model to the hyperparameters and the run folder, works as in [Train a model](training.md).

**What you'll learn:**

- How to convert tracking data into a Re-ID dataset
- How to set the length of an epoch for tracking crops
- How to sample the batches to train a model for tracking
- How to evaluate your model

---

## Install

```bash
pip install reid
```

---

## Quickstart

Cut the crops from videos with ground-truth tracks, then train with fixed-length epochs, batches from one sequence and re-association validation.

```python
from reid import ReassociationProtocol, ReIDModel, load_identity_folders
from reid.data.mot import generate_mot_patches

generate_mot_patches(mot_root="/path/to/sportsmot/train", output_dir="/path/to/train", split="full")
generate_mot_patches(mot_root="/path/to/sportsmot/val", output_dir="/path/to/val", split="full")
train_crops = load_identity_folders("/path/to/train")
val_crops = load_identity_folders("/path/to/val")

model = ReIDModel.from_pretrained("osnet_x1_0_msmt17_combineall")
history = model.train(
    train_crops,
    output_dir="runs/sportsmot",
    val_data=val_crops,
    val_protocol=ReassociationProtocol(),
    epochs=20,
    steps_per_epoch=1000,
    sample_per_sequence=True,
)
```

## Convert tracking data into a Re-ID dataset

Starting from your object tracking dataset in the [MOT format](https://trackers.roboflow.com/develop/evaluations/download/#output), you can use `generate_mot_patches` to turn it into a Re-ID dataset. It crops every box into one folder per track, named `<sequence>_<track_id>`, with one crop per frame. `load_identity_folders` then reads the folders back with one identity per track.

```python
from reid import load_identity_folders
from reid.data.mot import generate_mot_patches

generate_mot_patches(mot_root="/path/to/sportsmot/train", output_dir="/path/to/train", split="full")
train_crops = load_identity_folders("/path/to/train")
```

`split="full"` crops every frame of each sequence.

Split the training, validation and test sets by video. Clips of one video show the same people, often under different track IDs, so a model validated on another clip of a training video is scored on people it already learned, and the score would say little about new footage.

## Set the epoch length

An epoch draws about as many crops as the set holds. On tracking crops that makes epochs long for little gain, since every object is cropped in every frame and neighboring crops of a track are nearly the same image. SportsMOT train alone holds 312,579 crops. `steps_per_epoch` sets an epoch to a fixed number of batches instead:

```python
history = model.train(train_crops, val_data=val_crops, epochs=20, steps_per_epoch=1000)
```

Each epoch then draws 1000 batches of 32 crops, 32,000 crops in all, and the model is validated after each one. 20 such epochs see about twice as many crops as SportsMOT train holds and took about an hour on a Colab T4. 20 epochs that each pass over all 312,579 crops would take about 10 hours, and would validate the model only once every half hour.

## Draw each batch from one sequence

The triplet loss learns to separate each identity from the others in its batch, so who shares a batch decides which differences the model learns. By default the identities of a batch come from the whole set, which causes two problems on tracking crops.

**A track ID only holds inside its sequence.** Datasets like SportsMOT and DanceTrack number the tracks of each clip on their own, so one player seen in two clips of the same game has two IDs. A batch can hold both, and the loss then pushes the player away from themself.

**The objects a tracker confuses rarely share a batch.** A tracker has to tell apart the objects in one scene, such as teammates in the same uniform under the same light and camera. With many sequences to draw from, a batch rarely holds two objects from the same one, so it mostly asks the model to separate players from different games, who are easy to tell apart by their uniform or the court.

To solve these problems, turn on `sample_per_sequence=True`, which builds each batch from one selected sequence, so every identity a crop is separated from is one the tracker would really see in practice. Use it in `train` like this:

```python
history = model.train(train_crops, val_data=val_crops, sample_per_sequence=True)
```

The sequence of each crop is read from the paths `generate_mot_patches` writes. A sequence with fewer identities than `identities_per_batch` gives smaller batches, and one with a single identity is left out, since it has no one to separate.

This guide covers tracking with one camera, where each sequence numbers its own tracks. In multi-camera tracking, one ID follows a person across cameras, and `generate_mot_patches`, which names each identity after its sequence, would split that person into one identity per camera. Training for it needs a dataset built for it, with one identity per person across every camera.

> **Result placeholder.** The study's `per_sequence_batches` against `mixed_batches`: HOTA and IDF1 on SportsMOT val and test, each at the thresholds it picked on val. Write the study's statement in its exact wording, "one seed" included, and nothing it did not measure.

## Validate and read the training output

To validate a model for tracking, score it the way a tracker uses it. A tracker links an object to one of the boxes in the frames that follow, so `ReassociationProtocol` takes a box and ranks the boxes of the same sequence a set number of frames later by how close their embeddings are. `rank1` is the percentage of boxes whose closest match is the right one, and `mrr` averages 1 / rank of the right one. Pass it as `val_protocol`, and `train` keeps the epoch with the highest MRR in `best/`:

```python
from reid import ReassociationProtocol

history = model.train(train_crops, val_data=val_crops, val_protocol=ReassociationProtocol())
```

`train` prints the scores before training, then one line per epoch with its learning rate, losses and validation scores, and ends with the epoch it saved to `best/`. Fine-tuning `osnet_x1_0_msmt17_combineall` on SportsMOT train with the Quickstart's call prints:

> **Output placeholder.** The study's `per_sequence_batches` run's lines, verbatim: before training, epochs 1 to 4, the last epochs and the best-epoch line, with `...` between.

To plot them, use `plot_training_history`:

```python
from reid import plot_training_history

fig = plot_training_history("runs/sportsmot")
```

> **Figure placeholder.** `plot_training_history` on the same run's output folder.

<!-- ANALYSIS: what the curves show, read off the run's metrics.csv: the frozen epochs, the jump when the backbone unfreezes, the learning-rate drops and the best epoch. Numbers only from the run. -->

## Watch the embeddings separate

On the SportsMOT run, the embeddings plots of the 10 validation players with the most crops show each player's crops gathering into a cluster of their own:

> **Figure placeholder.** The same run's `embeddings/` plots: before training, an epoch while the backbone is frozen, the first epoch after it unfreezes, and the last epoch.

<!-- ANALYSIS: what the four panels show, read off the real images. If the plots don't show the players gathering, rewrite the sentence above to what they do show. -->

## Results

We tracked the SportsMOT test split with BoT-SORT without Re-ID and with each fine-tuned model, using the same YOLOX detections and motion parameters. Each model was trained on SportsMOT train, and its epoch was picked by MRR on the val split. BoT-SORT's Re-ID has two thresholds, `appearance_threshold` and `proximity_threshold`. For each model we tracked the val split with the same six combinations of them, built around the values the [trackers Re-ID guide](https://trackers.roboflow.com/develop/guides/reid/#results) tuned for SportsMOT, and kept the one with the highest HOTA, so each model is tested at the thresholds that suit it. Each model was then tracked once on the test split with its thresholds.

> **Table placeholder.** The study's Codabench test submissions, in the trackers Re-ID guide format.

<!-- TABLE: | Model | Batches | HOTA | IDF1 | MOTA |, rows: no Re-ID (the published geometry row 74.1 / 74.1 / 96.9, confirmed under trackers 2.7.0), osnet_x1_0_msmt17_combineall fine-tuned with batches from the whole set, and with per-sequence batches. Best value in bold. osnet_x1_0_sportsmot (77.8 HOTA, earlier trainer) only as a labelled reference if the study's control 2 reproduces it on this stack. Collapsible "Training and tracking configuration": each model's train() call, best epoch, and val-picked appearance_threshold and proximity_threshold, as the trackers guide's "Tuned ReID configuration" block. Motion parameters cite the trackers tracker comparison. -->
