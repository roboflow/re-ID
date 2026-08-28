---
title: Evaluate a query and gallery
---

# Evaluate a gallery

`ReIDModel.evaluate` computes re-identification metrics. Market-1501 and MSMT17 define their own query/gallery splits, so pass those explicitly; for a folder of identity crops, hand it a `ReIDDataset` and the partition is derived for you.

```python
from reid import ReIDModel, load_market1501

model = ReIDModel.from_pretrained()
query, gallery = load_market1501("/path/to/Market-1501")
result = model.evaluate(query, gallery)

print(result.metrics)
```

For a folder of identity crops, hand `evaluate` the dataset and the query/gallery partition is derived for you — tune it with `queries_per_id`, `min_crops`, and `seed`, or build it yourself and pass both splits:

```python
from reid import ReIDDataset

data = ReIDDataset("/path/to/identity/folders")

model.evaluate(data)  # defaults handle the split
model.evaluate(data, queries_per_id=2, min_crops=3)  # same, tuned
model.evaluate(*data.for_retrieval(seed=7))  # build it yourself
```

Passing an explicit gallery alongside a derivation argument raises, rather than silently ignoring it.

Evaluation uses cosine distance by default and returns the metrics with the evaluation output.
