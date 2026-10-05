---
title: Load a Re-ID model
---

# Load a model

`ReIDModel.from_pretrained()` loads a model from whichever source you have:

1. **Pretrained model name.** `ReIDModel.from_pretrained("osnet_x1_0_msmt17_combineall")`. The package supplies the architecture, weights, and preprocessing settings, and downloads the weights the first time a name is used; `list_aliases()` lists the supported names. `ReIDModel.from_pretrained()` with no name loads `osnet_x1_0_msmt17_combineall`.
2. **Saved model directory or Hugging Face repository.** `ReIDModel.from_pretrained("/path/to/export")` or `ReIDModel.from_pretrained("hf://org/repo")`. The directory must contain the weights and the `reid_config.json` that `save_pretrained` writes.
3. **Weights file.** `ReIDModel.from_pretrained("weights.pth", architecture="osnet_x1_0")`. Hugging Face (`hf://org/repo/file.pth`) and Google Drive (`gd://<file-id>`) sources work the same way. Name the architecture, because the file carries no model configuration; `list_architectures()` lists the options. If most of the architecture's weights are missing from the file, `from_pretrained` raises an error, because the weights come from a different model. A partial load above that, such as a checkpoint with a different head, loads with a warning, and the missing parameters keep random values. So make sure the weights file was trained with the architecture you name.
4. **Random initialization.** `ReIDModel.from_pretrained(architecture="osnet_x1_0")`. The weights are random, so the embeddings carry no identity information; use this only to test the API without downloading weights.

```python
from reid import ReIDModel

model = ReIDModel.from_pretrained("osnet_x1_0_msmt17_combineall")
model = ReIDModel.from_pretrained("/path/to/export")
model = ReIDModel.from_pretrained("weights.pth", architecture="osnet_x1_0")
```

`model.save_pretrained(directory)` writes the weights to `weights.safetensors` and the architecture and preprocessing settings to `reid_config.json`, so `ReIDModel.from_pretrained(directory)` restores the same model.

## Pretrained models

| Name                                     | Architecture             | Trained on                          | Weights from                                                                                                          |
| :--------------------------------------- | :----------------------- | :---------------------------------- | :-------------------------------------------------------------------------------------------------------------------- |
| `osnet_x1_0_msmt17_combineall` (default) | OSNet x1.0               | MSMT17, all splits combined         | [torchreid model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO), via Hugging Face `kaiyangzhou/osnet` |
| `fastreid_mot17_sbs50`                   | FastReID SBS, ResNeSt-50 | MOT17 train-half ground-truth crops | [BoT-SORT](https://github.com/NirAharon/BoT-SORT) release, via Google Drive                                           |

Weights download the first time a model is loaded and are kept on disk, so later loads skip the download; [Weight cache](#weight-cache) says where.

## Preprocessing

Each architecture has default preprocessing: OSNet resizes crops to 256 × 128 and FastReID SBS to 384 × 128, both by stretching. Pass a `ReIDPreprocessing` to `from_pretrained(..., preprocessing=...)` to override it. These are its fields, and what `reid_config.json` stores under `"preprocessing"`:

| Field | Meaning | Typical |
| :-- | :-- | :-- |
| `input_size` | Image height and width | OSNet `(256, 128)`; FastReID `(384, 128)` |
| `resize_mode` | `stretch` or `letterbox` | `stretch` |
| `interpolation` | Image resizing method | `bilinear` |
| `to_rgb` | Swap BGR to RGB for crops cut from video frames; crops read from disk are already RGB | `True` |
| `mean` / `std` | Per-channel normalization | ImageNet values |
| `pad_value` | Padding pixel value for `letterbox` | `114` |

`extract_embeddings` returns raw embedding vectors; evaluation normalizes them when it computes cosine distance.

## Weight cache

Google Drive downloads (`gd://...`) are cached under `~/.cache/reid/weights/`. Set `REID_CACHE_DIR` to move the cache; weights then go to `$REID_CACHE_DIR/weights/`. Hugging Face weights use the standard Hugging Face Hub cache.
