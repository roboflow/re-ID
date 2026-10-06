---
title: Re-ID documentation
description: Tools to extract and evaluate re-identification embeddings.
---

# Re-ID

Re-ID provides tools to load and evaluate re-identification models. Extract appearance embeddings from cropped images or detection boxes, and compare model performance with retrieval and re-association protocols.

Person re-identification (re-ID) recognizes the same person in different images: in another camera, or later in a video after they were out of view. A re-ID model turns each crop of a person into an embedding, a vector of numbers. Crops of the same person get embeddings that are close together and crops of different people get embeddings far apart, so the distance between two embeddings tells how likely the two crops show the same person.

<video width="100%" style="aspect-ratio: 16/9;" controls muted loop preload="metadata" aria-label="Gallery retrieval with the default reid model">
  <source src="https://github.com/user-attachments/assets/036a7949-2206-49fc-9a1b-89f9d3c51f20" type="video/mp4">
</video>

<p align="center" style="margin-top: -0.4em;"><small>Gallery retrieval with the default model, <code>osnet_x1_0_msmt17_combineall</code>: crops from other clips ranked by cosine distance to a query player, closest first.</small></p>

---

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
- [Evaluate a model](guides/evaluation.md): how a query is ranked against a gallery, and how to measure a model's accuracy
- [ReID metrics](learn/metrics.md): understand Rank-k, mAP, mINP, and MRR
