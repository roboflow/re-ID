---
title: Public API
---

# Public API

Import the public interface from `reid`:

```python
from reid import (
    ReIDDataset,
    ReIDModel,
    ReIDPreprocessing,
    ReIDSplit,
    ReassociationProtocol,
    RetrievalProtocol,
    list_aliases,
    list_architectures,
    load_market1501,
    load_msmt17,
)
```

`ReIDModel` provides methods to load a model (`from_pretrained`), measure identity-matching performance (`evaluate`), extract appearance embeddings (`extract_embeddings`), move it to a device (`to`), and save it (`save_pretrained`).

`ReIDDataset` loads images arranged as `<root>/<identity>/<crop>.jpg`. `ReIDSplit` holds image paths, person IDs, and camera IDs; benchmark loaders return separate query and gallery splits.

`ReIDModel.evaluate(data, protocol=...)` takes a protocol: `RetrievalProtocol` for retrieval or `ReassociationProtocol` for matching identities across a specified frame gap. Selection is deterministic, so evaluating several models with the same data and protocol scores identical inputs.

To compare a different set of images, subclass an `EvaluationProtocol`. Its `_build_query_gallery` returns the query split, the gallery split, an optional candidate mask, and a metadata dictionary that `evaluate` adds to `result.metadata`; its `metric_values` picks the metrics to report. The [evaluation guide](../guides/evaluation.md#limit-which-gallery-images-each-query-is-ranked-against) walks through an example. Import `EvaluationProtocol` and `ReIDMetrics` from `reid.eval`.

Use `ReIDModel.from_pretrained(architecture=...)` to initialize a model with random weights, or pass a checkpoint to load trained weights.

See the [loading guide](../guides/loading.md), [evaluation guide](../guides/evaluation.md), and [metrics guide](../learn/metrics.md) for runnable examples.
