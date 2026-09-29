---
title: 'ReID Metrics: mAP, Rank-k, mINP | reid'
description: Understand Rank-k, mAP, mINP, and MRR and use them to evaluate ReID model performance.
---

# ReID Metrics

ReID metrics measure how well a model finds images of the same person among candidate images. Standard retrieval benchmarks such as Market-1501 and MSMT17 test matching across cameras. Re-association evaluation tests matching within a sequence across a specified frame gap.

**What you'll learn:**

- What Rank-k (CMC), mAP, and mINP measure
- How Market-1501 / MSMT17 filtering (same camera, junk IDs) works
- How to evaluate a model with `ReIDModel.evaluate`
- How MRR measures re-association performance
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

Evaluate a pretrained model on the Market-1501 query and gallery splits:

```python
from reid import ReIDModel, load_market1501

model = ReIDModel.from_pretrained()
query, gallery = load_market1501("/path/to/Market-1501-v15.09.15")
result = model.evaluate(query, gallery)
print(result.metrics)
print(result.num_queries)
```

Metrics are percentages. A typical result looks like:

```text
{'map': ..., 'rank1': ..., 'rank5': ..., 'rank10': ..., 'minp': ...}
3368
```

The default alias (`osnet_x1_0_msmt17_combineall`) is trained on MSMT17 with train and test combined, so Market-1501 scores here measure **cross-domain** transfer (Rank-1 around 61 in [`notebooks/eval_reid.ipynb`](https://github.com/roboflow/re-ID/blob/main/notebooks/eval_reid.ipynb)). To match the torchreid OSNet x1.0 **same-domain** Market-1501 benchmark results (Rank-1 94.2, mAP 82.6, euclidean), load that checkpoint with `ReIDModel.from_pretrained(path, architecture="osnet_x1_0")`. Full Market-1501 and MSMT17 evaluation examples are in the same notebook.

---

## The retrieval setup

In standard Market-1501 and MSMT17 evaluation, the model ranks **gallery** images by their similarity to each **query** image. A valid match shows the same person from a different camera.

| Term    | Meaning                                                         |
| :------ | :-------------------------------------------------------------- |
| Query   | Image of the person you want to find                            |
| Gallery | Candidate crops to search                                       |
| Match   | Same person ID, different camera ID                             |
| Junk    | Same person and same camera, or labeled distractors (see below) |

Junk entries are removed from the ranked list before scoring. They never count as hits or as distractors that inflate rank.

**Same-(pid, camid) junk.** Standard single-query Market / MSMT protocol: a gallery crop of the same person from the same camera is discarded. Cross-camera retrieval is the goal.

**Gallery junk person IDs.** Market-1501 marks background distractors as `pid=0` and junk detections as `pid=-1`. `load_market1501` sets `gallery_junk_pids={-1, 0}` so both are ignored during ranking. Datasets where `pid=0` is a real identity keep the default `{-1}` only.

---

## Metrics

All metrics below are computed after junk filtering. Reported values are means over queries that have at least one valid gallery match, then scaled to percentages.

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
| Rank-5 / Rank-10 | Early list quality   | CMC curves                                  |
| mAP              | Full match set       | Primary gallery benchmark score             |
| mINP             | Hardest (last) match | Complements mAP when hard positives matter  |

---

## Distance metrics

`RetrievalProtocol(distance=...)` chooses how the model’s embeddings are compared to rank candidate matches.

| `distance`           | Behavior                                        |
| :------------------- | :---------------------------------------------- |
| `"cosine"` (default) | Cosine distance after L2-normalizing embeddings |
| `"euclidean"`        | Euclidean distance on the raw embedding vectors |

The torchreid model zoo reports OSNet x1.0 same-domain numbers with **euclidean** distance (94.2 / 82.6 on Market-1501). Match whatever distance a published table used before comparing, and do not mix distance choices across runs you want to compare.

```python
from reid.eval import RetrievalProtocol

result = model.evaluate(query, gallery, protocol=RetrievalProtocol(distance="euclidean"))
print(result.metrics)
```

---

## MSMT17

To evaluate on MSMT17, load its query and gallery splits:

```python
from reid import ReIDModel, load_msmt17

model = ReIDModel.from_pretrained("/path/to/osnet_x1_0_msmt17.pth", architecture="osnet_x1_0")
query, gallery = load_msmt17("/path/to/MSMT17_V1")
result = model.evaluate(query, gallery, max_pairs=1_000_000_000)
print(result.metrics)
```

The official test split compares 11,659 queries with 82,161 gallery images: about 958 million pairs and a 3.8 GB distance matrix, above the default `max_pairs` of 100 million.

Use a checkpoint trained for the dataset you evaluate. The default `combineall` MSMT17 weight includes MSMT17 test identities, so evaluating it on MSMT17 test leaks the split. Market-trained weights on MSMT17 (or the reverse) measure cross-domain transfer, rather than performance within the training domain.

---

## Re-association metrics

When using ReID for object tracking with [Trackers](https://github.com/roboflow/trackers), measure your model the way the tracker will use it. For short-term association, randomly picking frames throughout the whole video might not reflect how your ReID model is queried during real tracking.

`ReassociationProtocol` compares a query image with the people present at `query_frame + gap_frames` in the same sequence. It reports Rank-1 and mean reciprocal rank (`mrr`), both as percentages.

MRR measures how highly the model ranks the correct identity: first place contributes 1, second place contributes 1/2, and so on. Average these values over scored queries and multiply by 100. Each scored query has exactly one correct match, so its average precision equals its reciprocal rank.

For protocol configuration and the distinction from running a tracker, see the [evaluation guide](../guides/evaluation.md). Measure tracking outcomes separately with [Trackers](https://github.com/roboflow/trackers).

---

## Reference

- Zheng, L., et al. (2015). Scalable Person Re-identification: A Benchmark. ICCV. [Market-1501](https://zheng-lab-anu.github.io/Project/project_reid.html)
- Ye, M., et al. (2021). Deep Learning for Person Re-identification: A Survey and Outlook. IEEE TPAMI. (mINP)
- Zhou, K., et al. (2019). Omni-Scale Feature Learning for Person Re-Identification. ICCV. [arXiv:1905.00953](https://arxiv.org/abs/1905.00953) (OSNet; torchreid zoo baselines)
