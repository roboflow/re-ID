---
title: Load a Re-ID model
---

# Load a model

Use `ReIDModel.from_pretrained()` to load a pretrained model by name, a saved model directory, or a weights file.

```python
from reid import ReIDModel

# Load a supported pretrained model by name.
model = ReIDModel.from_pretrained("osnet_x1_0_msmt17_combineall")

# A weights file without model configuration also needs an architecture.
model = ReIDModel.from_pretrained("weights.pth", architecture="osnet_x1_0")
```

`model.save_pretrained(directory)` saves the weights and the configuration needed to reload the model, including its architecture and image preprocessing settings.

```python
model = ReIDModel.from_pretrained("/path/to/export")
```

Use `list_aliases()` and `list_architectures()` to discover the supported pretrained model names and architectures.
