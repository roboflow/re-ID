---
title: Train a Re-ID model
description: Fine-tune a Re-ID model on crops of your own objects, validate it during training, keep the best epoch, and watch the embeddings separate.
---

# Train a model

A pretrained model knows what people look like in general, but not the objects in your footage, which often look alike and appear under the same camera. Fine-tuning on crops from that footage teaches the model the differences that matter there, so a search finds the same object in your gallery, or a tracker keeps identities through crossings and occlusions. `ReIDModel.train` fine-tunes a model on crops whose identities you know, and given validation crops it scores the model after every epoch and keeps the epoch that scored highest. To train on crops cut from tracking video, read this guide first, then [Train a model for tracking](training-for-tracking.md).

**What you'll learn:**

- How to fine-tune a model on crops of known identities
- How training batches are sampled, and which hyperparameters to choose
- Why to start from a pretrained model
- How to read the training output and choose the best checkpoint

---

## Install

```bash
pip install reid
```

---

## Quickstart

Load the training and validation crops with `load_identity_folders`, start from a pretrained model, train with a validation set and an output folder, and load the best epoch when training ends.

```python
from reid import ReIDModel, load_identity_folders

train_crops = load_identity_folders("/path/to/train")
val_crops = load_identity_folders("/path/to/val")

model = ReIDModel.from_pretrained("osnet_x1_0_msmt17_combineall")
history = model.train(train_crops, output_dir="runs/my_run", val_data=val_crops)

best_model = ReIDModel.from_pretrained("runs/my_run/best")
```

The best epoch loads like any [saved model](loading.md), and `extract_embeddings` and `evaluate` work on it the same way.

## Prepare the crops

Load your crops with `load_identity_folders`, laid out as in [Evaluate a model](evaluation.md#evaluate-on-a-benchmark-or-your-own-images), or a benchmark's training split with its loader, such as `load_market1501_train` or `load_msmt17_train`. Keep apart a training set, a validation set and a test set, with no identity in more than one of them, so the scores measure the model on objects it never learned.

A benchmark often comes without a validation split, as Market-1501 does. To hold out every image of a number of people as a validation set, so no person is in both, use `train_val_split`:

```python
from reid import load_market1501_train

market_train = load_market1501_train("/path/to/Market-1501")
train_crops, val_crops = market_train.train_val_split(num_val_identities=100, seed=42)
```

## Sample the training batches

Each batch picks `identities_per_batch` different identities uniformly at random, then `crops_per_identity` crops of each, repeating crops when an identity has fewer. To learn correctly, the triplet loss needs to see other crops of the same identity in the batch, to pull them together, and crops of other identities, to push them apart. A batch of random crops from a large set would almost never hold two crops of the same identity.

```python
history = model.train(train_crops, val_data=val_crops, identities_per_batch=8, crops_per_identity=4)
```

More identities per batch give each crop more identities to be separated from, and more crops per identity show each one in more poses and light. Both cost memory, since a batch holds `identities_per_batch * crops_per_identity` crops.

An epoch is as many batches as the crops fill, so it draws about as many crops as the set holds. Within an epoch, no crop of an identity repeats until all of that identity's crops have been used. Since identities are drawn uniformly rather than by how many crops they have, an identity with many crops may not get through all of them in an epoch, and one with few might go through its crops more than once. Re-ID datasets are usually small enough for epochs of that size, such as the 12,936 training images of 751 people in Market-1501.

## Start from a pretrained model

Fine-tune a pretrained model rather than training one from scratch. Models like `osnet_x1_0_msmt17_combineall` have already learned what tells people apart, in its case from MSMT17's 126,441 images of 4,101 people, so training only has to adapt them to yours. It is one example, and a backbone pretrained on another dataset gives the same head start. Transfer learning pays off most with larger models trained on larger datasets.

```python
model = ReIDModel.from_pretrained("osnet_x1_0_msmt17_combineall")  # pretrained weights
model = ReIDModel("osnet_x1_0")  # random weights, to train from scratch
```

> **Figure placeholder.** Training loss by epoch for OSNet x1.0 started from `osnet_x1_0_msmt17_combineall` and from random weights, with the same recipe apart from the backbone freeze: the Market-1501 notebook's two runs, losses and validation mAP on shared axes.

<!-- ANALYSIS: what the two curves show, read off the two runs' metrics.csv. Write only what the runs show about how fast each converges and how close they end. -->

To let the new classification layer settle before it changes the pretrained features, `freeze_backbone_epochs` trains only the last layers for the first epochs. Training from scratch has no pretrained features to protect, so set `freeze_backbone_epochs=0`:

```python
model = ReIDModel("osnet_x1_0")
history = model.train(train_crops, val_data=val_crops, freeze_backbone_epochs=0)
```

## Choose the hyperparameters

The defaults train a pretrained OSNet well, so start with them and change one setting at a time, comparing runs on the validation score.

| Parameter                             | What it sets                                      | When to change it                                                             |
| :------------------------------------ | :------------------------------------------------ | :---------------------------------------------------------------------------- |
| `epochs`                              | Number of epochs                                  | Raise it while the validation score still rises at the end of a run           |
| `steps_per_epoch`                     | Batches per epoch                                 | Set it when an epoch over the whole set takes too long to validate often      |
| `lr`                                  | Learning rate after the warmup                    | Lower it when the loss jumps or validation drops early                        |
| `warmup_epochs`                       | Epochs the learning rate takes to reach `lr`      | Raise it when the first epochs are unstable                                   |
| `lr_scheduler`, `lr_scheduler_kwargs` | How the learning rate changes after the warmup    | Pass a scheduler from `torch.optim.lr_scheduler` to replace the default drops |
| `optimizer`, `optimizer_kwargs`       | The optimizer and its settings                    | Pass any `torch.optim` optimizer, such as `SGD` with `{"momentum": 0.9}`      |
| `weight_decay`                        | The optimizer's weight decay                      | Raise it when the model overfits the training identities                      |
| `freeze_backbone_epochs`              | Epochs that train only the last layers            | Set it to `0` to train from scratch                                           |
| `classification_loss`                 | Loss on the classification of training identities | Pass another loss, such as `CrossEntropyLoss` with other `label_smoothing`    |
| `embedding_loss`                      | Loss on the embeddings                            | Raise the `TripletLoss` margin to push identities further apart               |
| `classification_loss_weight`          | How much the classification loss counts           | Set it to `0` to train with the embedding loss alone                          |
| `embedding_loss_weight`               | How much the embedding loss counts                | Set it to `0` to train with the classification loss alone                     |
| `augmentations`                       | Random changes applied to each training crop      | Add or remove `torchvision.transforms` to match the variation in your footage |
| `num_workers`                         | Processes that load and augment crops             | Raise it when the GPU waits for data                                          |

After the warmup, the learning rate drops to a tenth three quarters of the way through the run and again at seven eighths, which is where the last gains in the [training output](#validate-and-read-the-training-output) come from. To use another schedule, pass the scheduler class and its arguments. `train` steps it once per epoch:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

history = model.train(train_crops, epochs=20, lr_scheduler=CosineAnnealingLR, lr_scheduler_kwargs={"T_max": 15})
```

## Validate and read the training output

With `val_data`, `train` scores the model with `evaluate` before training and after every epoch, and with `output_dir` it saves the epoch with the highest score to `best/`. By default it scores with `RetrievalProtocol`, as in [Evaluate a model](evaluation.md), and keeps the epoch with the highest mAP. `val_protocol` and `checkpoint_metric` choose another protocol or metric.

Each epoch prints one line with its learning rate, losses and validation scores, after a line with the scores before training, and the run ends with the epoch it saved to `best/`:

> **Output placeholder.** The Market-1501 run's lines, verbatim: before training, the first epochs, the last epochs and the best-epoch line, with `...` between.

The same rows come back from `train`. To plot them, from the returned list or later from a run's `metrics.csv`, use `plot_training_history`:

```python
from reid import plot_training_history

fig = plot_training_history(history)
fig = plot_training_history("runs/my_run")
```

> **Figure placeholder.** `plot_training_history` on the Market-1501 run, with the epoch saved to `best/` marked.

<!-- ANALYSIS: what the curves show, read off the Market-1501 run. -->

The model itself ends on the last epoch, so load `best/` to use the best one. Its validation score is the best of many tries on the same crops, so it overstates the picked epoch. Report the score on the test set, which neither training nor validation sees.

## Watch the embeddings separate

To see how training is going beyond the losses, look at a projection of the embeddings and check that the embeddings of each identity get closer together and form a cluster. With `val_data` and `output_dir`, `train` plots the embeddings of the `plot_identities` validation identities with the most crops, projected onto 2 dimensions with PCA, before training and after every epoch. The plots are saved in `embeddings/` in `output_dir`, and in a notebook each shows under its epoch's line.

> **Figure placeholder.** Panels from the Market-1501 run's `embeddings/`: before training, an early epoch, the epoch the backbone unfreezes, and the last epoch.

<!-- ANALYSIS: what the panels show, read off the real images. -->

PCA is fitted again for every plot, so compare how tight each color is and how far it sits from the others, not where it lands.

## The run folder

The run folder holds everything a run produces:

| Path                   | Contents                                                                                        |
| :--------------------- | :---------------------------------------------------------------------------------------------- |
| `last/`                | The model after the last epoch                                                                  |
| `best/`                | The epoch with the highest `checkpoint_metric` on `val_data`                                    |
| `metrics.csv`          | One row per epoch, the same rows `train` returns                                                |
| `training_config.json` | Every argument of the run, including the validation protocol and the metric that picked `best/` |
| `embeddings/`          | One embeddings plot per epoch                                                                   |

Both `last/` and `best/` load with `ReIDModel.from_pretrained`. So that a new run never overwrites an old one, `train` refuses an `output_dir` that already holds a run.

## Results

We scored `osnet_x1_0_msmt17_combineall`, a model pretrained on MSMT17, on Market-1501 as published and after fine-tuning it on the Market-1501 training split. Fine-tuning used `train`'s defaults, held out 100 of the 751 training people for validation, and kept the epoch with the highest validation mAP. Both were scored once on the official Market-1501 test split with `RetrievalProtocol` and cosine distance.

| Model                          | Pretrained on | Fine-tuned on |  Rank-1  |   mAP    |
| :----------------------------- | :------------ | :------------ | :------: | :------: |
| `osnet_x1_0_msmt17_combineall` | MSMT17        | none          |   66.5   |   40.0   |
| `osnet_x1_0_msmt17_combineall` | MSMT17        | Market-1501   | **93.9** | **85.0** |

The fine-tuned model performs much better on Market-1501 than the pretrained one, raising Rank-1 from 66.5 to 93.9 and mAP from 40.0 to 85.0.

??? info "Training and evaluation code"

    ```python
    from reid import ReIDModel, RetrievalProtocol, load_market1501_test, load_market1501_train

    market_train = load_market1501_train("/path/to/Market-1501")
    train_crops, val_crops = market_train.train_val_split(num_val_identities=100, seed=42)
    query, gallery = load_market1501_test("/path/to/Market-1501")

    model = ReIDModel.from_pretrained("osnet_x1_0_msmt17_combineall")
    model.train(train_crops, output_dir="runs/market1501", val_data=val_crops)

    best_model = ReIDModel.from_pretrained("runs/market1501/best")
    result = best_model.evaluate(query, gallery, protocol=RetrievalProtocol())
    ```

The fine-tuned model is one training run. Its best epoch, 39 of 40, scored within 0.1 of the last epoch on test.
