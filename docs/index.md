---
title: Re-ID documentation
description: Tools to extract and evaluate re-identification embeddings.
---

# Re-ID

Re-ID provides tools to load and evaluate re-identification models. Extract appearance embeddings from cropped images or detection boxes, and compare model performance with retrieval and re-association protocols.

## Install

```bash
pip install reid
```

## Quickstart

```python
import numpy as np
import supervision as sv
from reid import ReIDModel

# Random weights for this API example; load a pretrained model for actual matching.
model = ReIDModel.from_pretrained(architecture="osnet_x0_25", device="cpu")
frame = np.zeros((256, 128, 3), dtype=np.uint8)
detections = sv.Detections(xyxy=np.array([[0, 0, 128, 256]], dtype=np.float32))
embeddings = model.extract_embeddings(detections, frame)

print(embeddings.shape)  # (1, 512)
```

## Guides

- [Load a model](guides/loading.md): use pretrained weights or load a saved model
- [Evaluate a model](guides/evaluation.md): measure how accurately it matches identities
- [ReID metrics](learn/metrics.md): understand Rank-k, mAP, mINP, and MRR
