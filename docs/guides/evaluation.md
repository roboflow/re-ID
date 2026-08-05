---
title: Evaluate a query and gallery
---

# Evaluate a gallery

`ReIDEvaluator` computes re-identification metrics from query and gallery
splits. The built-in Market-1501 and MSMT17 loaders create those splits from
their dataset directories.

```python
from reid import ReIDEvaluator, ReIDModel, load_market1501

model = ReIDModel.from_pretrained()
query, gallery = load_market1501("/path/to/Market-1501")
result = ReIDEvaluator(model).evaluate(query, gallery)

print(result.metrics)
```

The evaluator uses cosine distance and returns the resulting metrics with the
evaluation output.
