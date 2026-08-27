---
title: Evaluate a query and gallery
---

# Evaluate a gallery

`ReIDModel.evaluate` computes re-identification metrics. Market-1501 and MSMT17 define their own query/gallery splits, so pass those explicitly; for a folder of identity crops, hand it a `ReIDCrops` and the partition is derived for you.

```python
from reid import ReIDModel, load_market1501

model = ReIDModel.from_pretrained()
query, gallery = load_market1501("/path/to/Market-1501")
result = model.evaluate(query, gallery)

print(result.metrics)
```

The evaluator uses cosine distance and returns the resulting metrics with the evaluation output.
