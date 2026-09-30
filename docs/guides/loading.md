---
title: Load a Re-ID model
---

# Load a model

`ReIDModel.from_pretrained()` loads a model from whichever source you have:

1. **Pretrained model name.** `ReIDModel.from_pretrained("osnet_x1_0_msmt17_combineall")`. Omitting the name loads this default model. The package supplies the architecture, weights, and preprocessing settings; `list_aliases()` lists the supported names.
2. **Saved model directory or Hugging Face repository.** `ReIDModel.from_pretrained("/path/to/export")` or `ReIDModel.from_pretrained("hf://org/repo")`. The directory must contain the weights and the `reid_config.json` that `save_pretrained` writes.
3. **Weights file.** `ReIDModel.from_pretrained("weights.pth", architecture="osnet_x1_0")`. Name the architecture, because the file carries no model configuration; `list_architectures()` lists the options. Hugging Face (`hf://org/repo/file.pth`) and Google Drive (`gd://<file-id>`) sources work the same way.
4. **Random initialization.** `ReIDModel.from_pretrained(architecture="osnet_x1_0")`. Use this to test the API without downloading weights.

```python
from reid import ReIDModel

model = ReIDModel.from_pretrained("osnet_x1_0_msmt17_combineall")
model = ReIDModel.from_pretrained("/path/to/export")
model = ReIDModel.from_pretrained("weights.pth", architecture="osnet_x1_0")
```

`model.save_pretrained(directory)` writes the weights to `weights.safetensors` and the architecture and preprocessing settings to `reid_config.json`, so `ReIDModel.from_pretrained(directory)` restores the same model.

## Preprocessing

Each architecture has default preprocessing: OSNet resizes crops to 256 × 128 and FastReID SBS to 384 × 128, both by stretching. Pass a `ReIDPreprocessing` to `from_pretrained(..., preprocessing=...)` to override it. These are its fields, and what `reid_config.json` stores under `"preprocessing"`:

| Field | Meaning | Typical |
| :-- | :-- | :-- |
| `input_size` | Image height and width | OSNet `(256, 128)`; FastReID `(384, 128)` |
| `resize_mode` | `stretch` or `letterbox` | `stretch` |
| `interpolation` | Image resizing method | `bilinear` |
| `to_rgb` | Swap BGR to RGB for crops cut from video frames; crops read from disk are already RGB | `True` |
| `mean` / `std` | Per-channel normalization | ImageNet |
| `pad_value` | Padding pixel value for `letterbox` | `114` |

`extract_embeddings` returns raw embedding vectors; evaluation normalizes them when it computes cosine distance.

## Weight cache

Google Drive downloads (`gd://...`) are cached under `~/.cache/reid/weights/`. Set `REID_CACHE_DIR` to move the cache; weights then go to `$REID_CACHE_DIR/weights/`. Hugging Face weights use the standard Hugging Face Hub cache.
