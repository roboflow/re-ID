---
title: ReID Metrics — mAP, Rank-k, mINP | reid
description: Learn gallery person re-identification metrics — CMC Rank-k, mAP, and mINP — and how to evaluate embeddings with reid on Market-1501 and MSMT17.
---

# ReID Metrics

Gallery re-identification scores how well an appearance encoder ranks the right identity among many candidates. Each query image is matched against a gallery of crops from other cameras. That is a different task from multi-object tracking metrics (HOTA, IDF1, MOTA), which [Roboflow Trackers](https://github.com/roboflow/trackers) reports on MOT benchmarks.

**What you'll learn:**

- What Rank-k (CMC), mAP, and mINP measure
- How Market-1501 / MSMT17 filtering (same camera, junk IDs) works
- How to run gallery evaluation with `ReIDModel.evaluate`
- When to use cosine vs Euclidean distance

---

## Install

```bash
pip install reid
```

For local development:

```bash
pip install -e ".[dev]"
```

---

## Quickstart

Load Market-1501 query and gallery splits, embed them with a pretrained model, and report metrics:

```python
from reid import ReIDModel, load_market1501

model = ReIDModel.from_pretrained()
query, gallery = load_market1501("/path/to/Market-1501-v15.09.15")
result = model.evaluate(query, gallery)
print(result.metrics)
```

Scores are percentages. A typical printout looks like:

```text
mAP: ...%  Rank-1: ...%  Rank-5: ...%  Rank-10: ...%  mINP: ...%  (n=3368)
```

The default alias (`osnet_x1_0_msmt17_combineall`) is trained on MSMT17 with train and test combined, so Market-1501 scores here measure **cross-domain** transfer (Rank-1 around 61 in [`notebooks/eval_reid.ipynb`](https://github.com/roboflow/re-ID/blob/main/notebooks/eval_reid.ipynb)). To match the torchreid OSNet x1.0 **same-domain** Market-1501 zoo targets (Rank-1 94.2, mAP 82.6, euclidean), load that checkpoint with `ReIDModel.from_pretrained(path, architecture="osnet_x1_0")`. Full Market-1501 and MSMT17 recipes are in the same notebook.

---

## The retrieval setup

For each **query** image you sort the **gallery** by embedding distance (lower is closer). A gallery image is a **match** when it shares the query person ID and comes from a **different** camera.

| Term    | Meaning                                                         |
| :------ | :-------------------------------------------------------------- |
| Query   | Probe crop whose identity you want to find                      |
| Gallery | Candidate crops to search                                       |
| Match   | Same person ID, different camera ID                             |
| Junk    | Same person and same camera, or labeled distractors (see below) |

Junk entries are removed from the ranked list before scoring. They never count as hits or as distractors that inflate rank.

**Same-(pid, camid) junk.** Standard single-query Market / MSMT protocol: a gallery crop of the same person from the same camera is discarded. Cross-camera retrieval is the goal.

**Gallery junk person IDs.** Market-1501 marks background distractors as `pid=0` and junk detections as `pid=-1`. `load_market1501` sets `gallery_junk_pids={-1, 0}` so both are ignored during ranking. Datasets where `pid=0` is a real identity keep the default `{-1}` only.

---

## Metrics

All metrics below are computed by `reid.eval.metrics.compute_reid_metrics` after junk filtering. Reported values are means over queries that have at least one valid gallery match, then scaled to percentages.

### Rank-k (CMC)

**Cumulative Matching Characteristic.** Rank-k is the fraction of queries where at least one true match appears in the top-k of the filtered gallery ranking.

- **Rank-1:** Is the nearest neighbor the correct identity?
- **Rank-5 / Rank-10:** Is a correct identity among the first 5 or 10?

CMC answers "how often is a hit early in the list?" It does not reward finding every gallery instance of that person.

### mAP

**Mean Average Precision.** For each query, average precision (AP) integrates precision at every rank where a true match occurs. mAP is the mean of those APs over queries.

Unlike Rank-1, mAP cares about **all** valid gallery matches for that identity, not only the first hit. It is the usual primary number on Market-1501 and MSMT17 leaderboards.

### mINP

**Mean Inverse Negative Penalty** (Ye et al., TPAMI 2021). For each query, INP is `N_rel / R_last`, where `N_rel` is the number of valid matches and `R_last` is the rank of the **last** true match. mINP averages INP over queries.

mINP penalizes models that bury hard positive gallery images deep in the ranking even when Rank-1 and mAP look strong.

| Metric           | Sensitive to         | Typical use                                 |
| :--------------- | :------------------- | :------------------------------------------ |
| Rank-1           | First hit only       | Quick sanity check; often reported with mAP |
| Rank-5 / Rank-10 | Early list quality   | CMC curves; `max_rank` must be at least 10  |
| mAP              | Full match set       | Primary gallery benchmark score             |
| mINP             | Hardest (last) match | Complements mAP when hard positives matter  |

---

## Distance metrics

`model.evaluate(..., distance=...)` builds a query × gallery distance matrix before calling the metric functions.

| `distance`           | Behavior                                        |
| :------------------- | :---------------------------------------------- |
| `"cosine"` (default) | Cosine distance after L2-normalizing embeddings |
| `"euclidean"`        | Euclidean distance on the raw embedding vectors |

The torchreid model zoo reports OSNet x1.0 same-domain numbers with **euclidean** distance (94.2 / 82.6 on Market-1501). Match whatever distance a published table used before comparing, and do not mix distance choices across runs you want to compare.

---

## MSMT17

Same metric stack, different loader:

```python
from reid import ReIDModel, load_msmt17

model = ReIDModel.from_pretrained("/path/to/osnet_x1_0_msmt17.pth", architecture="osnet_x1_0")
query, gallery = load_msmt17("/path/to/MSMT17_V1")
result = model.evaluate(query, gallery, distance="cosine")
print(result.metrics)
```

Use a checkpoint trained for the dataset you evaluate. The default `combineall` MSMT17 weight includes MSMT17 test identities, so evaluating it on MSMT17 test leaks the split. Market-trained weights on MSMT17 (or the reverse) measure cross-domain transfer, not same-domain zoo scores.

---

## Tracking appearance vs gallery metrics

BoT-SORT in [trackers](https://github.com/roboflow/trackers) uses cosine similarity between live detections and a per-track feature bank. That can improve MOT ID metrics on some scenes.

Gallery mAP / Rank-k answer a different question: how good is the embedding space for offline retrieval on person ReID benchmarks. Train or select encoders with gallery metrics; measure tracking impact with MOT evaluation in trackers.

### MOT17 fine-tune protocol (tracking-oriented)

The MOT17 notebook fine-tunes on **train-half** GT crops and scores retrieval on **val-half** GT crops of the **same sequences**. Frames are temporally disjoint; person IDs, cameras, and scenes can overlap. That is intentional when the goal is adapting appearance features for BoT-SORT on MOT17 val, not claiming open-set ReID gains like Market-1501.

Under that protocol:

- Retrieval mAP / Rank-1 are a **closed-scene temporal diagnostic** (easy near-duplicate gallery hits are possible).
- Tracking HOTA / IDF1 on val-half are the primary success signal.
- For identity-disjoint crop experiments, use `build_identity_crop_holdout_split` instead of evaluating all val identities.

---

## Reference

- Zheng, L., et al. (2015). Scalable Person Re-identification: A Benchmark. ICCV. [Market-1501](https://zheng-lab-anu.github.io/Project/project_reid.html)
- Ye, M., et al. (2021). Deep Learning for Person Re-identification: A Survey and Outlook. IEEE TPAMI. (mINP)
- Zhou, K., et al. (2019). Omni-Scale Feature Learning for Person Re-Identification. ICCV. [arXiv:1905.00953](https://arxiv.org/abs/1905.00953) (OSNet; torchreid zoo baselines)
