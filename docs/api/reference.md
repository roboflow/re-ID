---
title: Public API
---

# Public API

Import the public interface from `reid`:

```python
from reid import (
    ReIDCrops,
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

`ReIDModel` carries the whole lifecycle: `from_pretrained` to load, `train` to fine-tune in place, `evaluate` to score retrieval, `extract_features` for inference, `to` to move devices, and `save_pretrained` to persist. `ReIDCrops` wraps a `<root>/<identity>/<crop>.jpg` folder and hands out the partition each step needs; `ReIDSplit` is the underlying paths/pids/camids record that `load_market1501` and `load_msmt17` return.

For a training loop you drive yourself, `reid.training` exposes `train_reid` and `TrainConfig`, and `ReIDEvaluator` scores an explicit query/gallery pair.

See the [loading guide](../guides/loading.md), [evaluation guide](../guides/evaluation.md), and [fine-tuning guide](../learn/train.md) for runnable examples.
