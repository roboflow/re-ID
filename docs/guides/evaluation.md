---
title: Evaluate a ReID model
---

# Evaluate a model

`ReIDModel.evaluate` measures how accurately a model matches identities. Retrieval compares each query image with a gallery of candidate images and ranks them by appearance similarity. Market-1501 and MSMT17 provide their own query and gallery splits; pass these directly to the model.

```python
from reid import ReIDModel, load_market1501

model = ReIDModel.from_pretrained()
query, gallery = load_market1501("/path/to/Market-1501")
result = model.evaluate(query, gallery)

print(result.metrics)
print(result.num_queries)
```

Your own images usually come without such splits. If they are in identity folders, one per person (`<root>/<identity>/<image>.jpg`), `RetrievalProtocol` builds the split: it picks `queries_per_id` random images of each person as queries, using `seed`, and puts the rest in the gallery, so every query has a correct match to find. People with fewer than `min_crops` images are left out:

```python
from reid import ReIDDataset
from reid.eval import RetrievalProtocol

data = ReIDDataset("/path/to/identity/folders")

result = model.evaluate(data, protocol=RetrievalProtocol(queries_per_id=2, min_crops=3, seed=7))
```

Omit `protocol` to use retrieval with one query per identity. The same data and protocol, including its `seed`, always select the same images, so evaluating several models this way compares them on identical inputs. `RetrievalProtocol(...).build_query_gallery(data)` returns the query and gallery images it evaluates, the candidate mask (`None` for retrieval), and the metadata. To compare query and gallery images you chose yourself, build a `ReIDSplit` for each and pass both to `model.evaluate`.

`result.metrics` is a dictionary with metric names as keys and percentages as values. For retrieval, `map` (mean average precision) measures ranking across all correct matches, `rank1`, `rank5`, and `rank10` report the percentage of queries with a correct match among the 1, 5, or 10 gallery images closest to the query, and `minp` (mean inverse negative penalty) reflects the position of the last correct match. Re-association reports `rank1` and `mrr` instead, described [below](#crops-from-video). [ReID metrics](../learn/metrics.md#metrics) gives each metric's formula. `result.num_queries` counts queries included in the metrics; `result.metadata` records the protocol settings and the number of query and gallery images.

Retrieval uses cosine distance by default; set `RetrievalProtocol(distance="euclidean")` when the benchmark specifies Euclidean distance. `max_pairs` limits the number of query–gallery comparisons to control memory use; a larger evaluation raises with its matrix size and memory, and MSMT17's official split needs `max_pairs=1_000_000_000`.

## Crops from video

Evaluate a model the way you will use it: the queries and gallery should look like the comparisons the model makes in your application. When re-ID helps a tracker such as [Roboflow Trackers](https://github.com/roboflow/trackers) associate people across frames, the model has to recognize a person again among the people visible a number of frames later. `ReassociationProtocol` measures that. For a person's crop at one frame, the query, it takes the people present in the same sequence a set number of frames later, ranks them by similarity to the query, and records the rank of that same person. Retrieval on the same crops would overstate this ability, because a query's closest gallery images are usually the same person a few frames apart.

`generate_mot_patches` builds its input from sequences with MOT-format ground truth (a `gt/gt.txt` file and an `img1/` frame folder per sequence): one folder per identity (`<sequence>_<track_id>`) with one crop per box, named by frame.

```python
from reid import ReIDDataset
from reid.data.mot import generate_mot_patches
from reid.eval import ReassociationProtocol

generate_mot_patches(mot_root="/path/to/dancetrack/val", output_dir="/path/to/crops", split="full")

crops = ReIDDataset("/path/to/crops")
result = model.evaluate(crops, protocol=ReassociationProtocol(gap_frames=30, query_stride=60))
print(result.metrics)
```

`split` chooses which frames of each sequence are cropped:

| `split` | Frames cropped |
| :-- | :-- |
| `"full"` | every frame |
| `"train_half"` (default) | the first half of each sequence |
| `"val_half"` | the second half of each sequence |

Use `"full"` for sequences kept only for evaluation. Use `"train_half"` and `"val_half"` when training and evaluating on the same sequences, so no frame is used for both. `class_ids` selects which ground-truth classes become identities. DanceTrack and SportsMOT label every box as class 1, so the default keeps all of them; MOT17 also labels vehicles, static people, distractors, and occluders, so pass `class_ids=[1]` there.

Re-association reports `rank1`, the percentage of queries whose correct match is the closest of their candidates, and `mrr`, the [mean reciprocal rank](../learn/metrics.md#mrr): each query scores 1/r, where r is the rank of its correct match (1 at rank 1, 1/2 at rank 2), and `mrr` averages these scores. Each query has exactly one correct match, so mean average precision is the mean reciprocal rank.

Re-association scores rankings of saved crops; it does not run a tracker or decide any match. Set `gap_frames` to the gaps your tracker has to bridge, and measure tracking performance separately with [Roboflow Trackers](https://github.com/roboflow/trackers).

## Limit which gallery images each query is ranked against

By default, each query is ranked against the whole gallery. A candidate set limits a query to some of the gallery images. `ReassociationProtocol` builds one for every query: with `gap_frames=30`, a query at frame 100 is ranked only against the people present at frame 130 of the same sequence.

To define your own candidate sets, subclass a protocol and return a candidate mask from `_build_query_gallery`. The mask is a boolean array shaped `(num_queries, num_gallery)`: `True` keeps a gallery image in that query's ranking, and `False` removes it from both the ranking and the count of correct matches. This protocol ranks each query only against gallery images from the same scene:

```python
import numpy as np

from reid.eval import RetrievalProtocol


class SameSceneRetrieval(RetrievalProtocol):
    def __init__(self, scene_of, **kwargs):
        super().__init__(**kwargs)
        self.scene_of = scene_of

    def _build_query_gallery(self, data, gallery=None, *, max_pairs):
        query, gallery, _, metadata = super()._build_query_gallery(data, gallery, max_pairs=max_pairs)
        query_scenes = np.array([self.scene_of[path] for path in query.image_paths])
        gallery_scenes = np.array([self.scene_of[path] for path in gallery.image_paths])
        return query, gallery, query_scenes[:, None] == gallery_scenes[None, :], metadata


result = model.evaluate(data, protocol=SameSceneRetrieval(scene_of))
```

Camera and junk-ID filtering still apply inside the candidate set. Set `exclude_same_pid_camid = False` on the class when same-person, same-camera matches should count, as in re-association. Queries with no correct match among their candidates are left out of the averages and of `num_queries`; if no query can be scored, `evaluate` raises a `ValueError`. The distance matrix is still computed in full, so `max_pairs` counts every query–gallery pair, including those outside the candidate sets.
