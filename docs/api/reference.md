---
title: Public API
---

# Public API

Import the public interface from `reid`:

```python
from reid import (
    ReIDEvaluator,
    ReIDModel,
    ReIDPreprocessing,
    compute_reid_metrics,
    list_aliases,
    list_architectures,
    load_market1501,
    load_msmt17,
)
```

`ReIDModel` extracts embeddings, `ReIDEvaluator` evaluates query/gallery splits, and `ReIDPreprocessing` records the model input recipe. See the [loading guide](../guides/loading.md) and [evaluation guide](../guides/evaluation.md) for runnable examples.
