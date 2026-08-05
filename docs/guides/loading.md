---
title: Load a Re-ID model
---

# Load a model

Use `ReIDModel.from_pretrained()` for curated aliases, saved exports, or bare
checkpoint files.

```python
from reid import ReIDModel

# Curated checkpoint alias.
model = ReIDModel.from_pretrained("osnet_x1_0_msmt17_combineall")

# A checkpoint needs its architecture declared.
model = ReIDModel.from_pretrained("weights.pth", architecture="osnet_x1_0")
```

`save_pretrained()` exports a `reid_config.json` alongside safetensors weights.
Load that directory directly to restore its recorded architecture and
preprocessing configuration.

```python
model = ReIDModel.from_pretrained("/path/to/export")
```

Use `list_aliases()` and `list_architectures()` to discover the currently
supported curated checkpoints and architectures.
