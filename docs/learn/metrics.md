---
title: 'ReID Metrics: mAP, Rank-k, mINP | reid'
description: Understand Rank-k, mAP, mINP, and MRR and use them to evaluate ReID model performance.
---

# ReID Metrics

ReID metrics measure how well a model finds images of the same person among candidate images. Standard retrieval benchmarks such as Market-1501 and MSMT17 test matching across cameras. Re-association evaluation tests matching within a sequence across a specified frame gap.

**What you'll learn:**

- How queries are compared with gallery images, and which gallery images are ignored
- What Rank-k (CMC), mAP, mINP, and MRR measure, and how each is computed
- How to evaluate a model with `ReIDModel.evaluate`
- When to use cosine vs Euclidean distance

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

The default model was trained on MSMT17, so these Market-1501 numbers measure cross-domain transfer; [Benchmarks](#benchmarks) shows how to reproduce the published same-domain results.

---

## Evaluation setup

### Retrieval

In standard [Market-1501](https://openaccess.thecvf.com/content_iccv_2015/html/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.html) and [MSMT17](https://arxiv.org/abs/1711.08565) evaluation, the model ranks **gallery** images by their distance to each **query** image. A valid match shows the same person from a different camera.

| Term    | Meaning                                                         |
| :------ | :-------------------------------------------------------------- |
| Query   | Image of the person you want to find                            |
| Gallery | Candidate crops to search                                       |
| Match   | Same person ID, different camera ID                             |
| Junk    | Same person and same camera, or labeled distractors (see below) |

Junk entries are removed from the ranked list before scoring. They never count as hits or as distractors that push a hit down the ranking.

**Same-(pid, camid) junk.** Standard single-query Market / MSMT protocol: a gallery crop of the same person from the same camera is discarded. Cross-camera retrieval is the goal.

**Gallery junk person IDs.** Market-1501 marks background distractors as `pid=0` and junk detections as `pid=-1`. `load_market1501` sets `gallery_junk_pids={-1, 0}` so both are ignored during ranking. Datasets where `pid=0` is a real identity keep the default `{-1}` only.

### Re-association

When using ReID for object tracking with [Trackers](https://github.com/roboflow/trackers), measure your model the way the tracker will use it. For short-term association, randomly picking frames throughout the whole video might not reflect how your ReID model is queried during real tracking.

`ReassociationProtocol` compares a query image only with the people present at `query_frame + gap_frames` in the same sequence, so each query has exactly one correct match among its candidates. For protocol configuration and the distinction from running a tracker, see the [evaluation guide](../guides/evaluation.md). Measure tracking outcomes separately with [Trackers](https://github.com/roboflow/trackers).

### Distance

`RetrievalProtocol(distance=...)` chooses how the model's embeddings are compared to rank candidate matches.

| `distance`           | Behavior                                        |
| :------------------- | :---------------------------------------------- |
| `"cosine"` (default) | Cosine distance after L2-normalizing embeddings |
| `"euclidean"`        | Euclidean distance on the raw embedding vectors |

The torchreid model zoo reports OSNet x1.0 same-domain numbers with **euclidean** distance (94.2 / 82.6 on Market-1501). Use the same distance as the published table you compare against, and do not mix distance choices across runs you want to compare.

```python
from reid.eval import RetrievalProtocol

result = model.evaluate(query, gallery, protocol=RetrievalProtocol(distance="euclidean"))
print(result.metrics)
```

---

## Metrics

For each query, the model compares the query's embedding with every gallery image's embedding using the chosen [distance](#distance), and sorts the gallery from closest to farthest. An image's **rank** is its position in that list: rank 1 is the closest.

Every metric is computed on each query's ranking after junk filtering, averaged over the queries that have at least one valid match, and reported as a percentage. The notation used below:

| Symbol | Meaning |
| :-- | :-- |
| $Q$ | Queries with at least one valid match |
| $\mathrm{rel}_q(i)$ | 1 if the gallery image at rank $i$ matches query $q$, otherwise 0 |
| $G_q$ | Number of valid matches for query $q$ |
| $P_q(i)$ | Precision at rank $i$: the fraction of the top $i$ that are matches |

### Rank-k (CMC)

**Cumulative Matching Characteristic**, not to be confused with camera motion compensation, which trackers such as BoT-SORT also abbreviate CMC. Rank-k is the fraction of queries with at least one match in the top $k$, as reported in the [Market-1501 benchmark](https://openaccess.thecvf.com/content_iccv_2015/html/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.html):

$$
\text{Rank-}k = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{1}\left[\, \sum_{i=1}^{k} \mathrm{rel}_q(i) \geq 1 \right]
$$

- **Rank-1:** Is the gallery image closest to the query the same person?
- **Rank-5 / Rank-10:** Is the same person among the 5 or 10 closest gallery images?

CMC answers "how often is a hit early in the list?" It does not reward finding every gallery instance of that person.

### mAP

**Mean Average Precision.** Average precision takes the precision at every rank where a match occurs, averaged over the query's matches; mAP averages it over queries. The [Market-1501 paper](https://openaccess.thecvf.com/content_iccv_2015/html/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.html) introduced it to re-identification, borrowing the [information-retrieval measure](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision):

$$
\mathrm{AP}_q = \frac{1}{G_q} \sum_{i} P_q(i)\, \mathrm{rel}_q(i), \qquad \mathrm{mAP} = \frac{1}{|Q|} \sum_{q \in Q} \mathrm{AP}_q
$$

Unlike Rank-1, mAP cares about **all** valid gallery matches for that identity, not only the first hit. It is the usual primary number on Market-1501 and MSMT17 leaderboards.

### mINP

**Mean Inverse Negative Penalty**, proposed by [Ye et al. (TPAMI 2021)](https://arxiv.org/abs/2001.04193). With $R_q$ being the rank of the **last** match of query $q$:

$$
\mathrm{INP}_q = \frac{G_q}{R_q}, \qquad \mathrm{mINP} = \frac{1}{|Q|} \sum_{q \in Q} \mathrm{INP}_q
$$

mINP penalizes models that bury hard positive gallery images deep in the ranking even when Rank-1 and mAP look strong.

### MRR

**Mean reciprocal rank** ([definition](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)) is what re-association reports next to Rank-1. With $r_q$ being the rank of the single correct match among the query's candidates:

$$
\mathrm{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{r_q}
$$

First place contributes 1, second place 1/2, and so on. With exactly one correct match per query, average precision equals the reciprocal rank, so MRR is the mAP of the re-association ranking.

### Comparing the metrics

| Metric           | Sensitive to         | Typical use                                 |
| :--------------- | :------------------- | :------------------------------------------ |
| Rank-1           | First hit only       | Quick sanity check; often reported with mAP |
| Rank-5 / Rank-10 | Early list quality   | CMC curves                                  |
| mAP              | Full match set       | Primary gallery benchmark score             |
| mINP             | Hardest (last) match | Complements mAP when hard positives matter  |
| MRR              | Rank of the one match | Re-association, one correct match per query |

---

## Benchmarks

Use a checkpoint trained on the training split of the dataset you evaluate. The default `osnet_x1_0_msmt17_combineall` was trained on all of MSMT17, test identities included: on MSMT17 it leaks the test split, and on Market-1501 it measures **cross-domain** transfer (Rank-1 around 61 in [`notebooks/eval_reid.ipynb`](https://github.com/roboflow/re-ID/blob/main/notebooks/eval_reid.ipynb)). The examples below load the torchreid same-domain OSNet x1.0 checkpoints and use **euclidean** distance, as the model zoo does; the notebook runs both end to end.

### Market-1501

```python
from reid import ReIDModel, load_market1501
from reid.eval import RetrievalProtocol

model = ReIDModel.from_pretrained("/path/to/osnet_x1_0_market1501.pth", architecture="osnet_x1_0")
query, gallery = load_market1501("/path/to/Market-1501-v15.09.15")
result = model.evaluate(query, gallery, protocol=RetrievalProtocol(distance="euclidean"))
print(result.metrics)
```

The model zoo reports Rank-1 94.2 and mAP 82.6 for this checkpoint.

### MSMT17

```python
from reid import ReIDModel, load_msmt17
from reid.eval import RetrievalProtocol

model = ReIDModel.from_pretrained("/path/to/osnet_x1_0_msmt17.pth", architecture="osnet_x1_0")
query, gallery = load_msmt17("/path/to/MSMT17_V1")
result = model.evaluate(query, gallery, protocol=RetrievalProtocol(distance="euclidean"))
print(result.metrics)
```

The model zoo reports Rank-1 74.9 and mAP 43.8 for this checkpoint. The official test split compares 11,659 queries with 82,161 gallery images: about 958 million pairs and a 3.8 GB distance matrix, so make sure the machine has that memory free.

---

## Reference

- Zheng, L., et al. (2015). Scalable Person Re-identification: A Benchmark. ICCV. [Paper](https://openaccess.thecvf.com/content_iccv_2015/html/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.html), [dataset](https://zheng-lab-anu.github.io/Project/project_reid.html)
- Wei, L., et al. (2018). Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR. [arXiv:1711.08565](https://arxiv.org/abs/1711.08565) (MSMT17)
- Ye, M., et al. (2021). Deep Learning for Person Re-identification: A Survey and Outlook. IEEE TPAMI. [arXiv:2001.04193](https://arxiv.org/abs/2001.04193) (mINP)
- Zhou, K., et al. (2019). Omni-Scale Feature Learning for Person Re-Identification. ICCV. [arXiv:1905.00953](https://arxiv.org/abs/1905.00953) (OSNet; torchreid zoo baselines)
