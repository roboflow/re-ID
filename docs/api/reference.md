---
title: Public API
---

# Public API

Import the public interface from `reid`:

```python
from reid import (
    ReIDDataset,
    ReIDEvaluator,
    ReIDModel,
    ReIDPreprocessing,
    ReIDSplit,
    compute_reid_metrics,
    list_aliases,
    list_architectures,
    load_market1501,
    load_msmt17,
)
```

`ReIDModel` carries the lifecycle: `from_pretrained` to load, `evaluate` to score retrieval, `extract_features` for inference, `to` to move devices, and `save_pretrained` to persist. `ReIDDataset` wraps a `<root>/<identity>/<crop>.jpg` folder and hands out the partition each step needs; `ReIDSplit` is the underlying paths/pids/camids record that `load_market1501` and `load_msmt17` return.

`ReIDEvaluator` scores an explicit query/gallery pair for callers who want it directly.

See the [loading guide](../guides/loading.md), [evaluation guide](../guides/evaluation.md), and [metrics guide](../learn/metrics.md) for runnable examples.
