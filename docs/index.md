---
title: Re-ID documentation
description: Re-identification models and gallery evaluation for appearance embeddings.
---

# Re-ID

Re-ID provides appearance-embedding models and gallery evaluation tools for person re-identification workflows.

## Install

```bash
uv sync --all-extras
```

## Quickstart

```python
import numpy as np
import supervision as sv
from reid import ReIDModel

model = ReIDModel.from_pretrained(architecture="osnet_x0_25", device="cpu")
frame = np.zeros((256, 128, 3), dtype=np.uint8)
detections = sv.Detections(xyxy=np.array([[0, 0, 128, 256]], dtype=np.float32))
embeddings = model.extract_features(detections, frame)

print(embeddings.shape)  # (1, 512)
```

Continue with the [model-loading guide](guides/loading.md) or see how to [evaluate a query/gallery split](guides/evaluation.md).
