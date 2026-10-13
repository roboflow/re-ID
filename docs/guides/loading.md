---
title: Load a Re-ID model
---

# Load a model

`ReIDModel.from_pretrained()` loads a model from whichever source you have:

1. **Pretrained model name.** `ReIDModel.from_pretrained("osnet_x1_0_msmt17_combineall")`. The package supplies the architecture, weights, and preprocessing settings, and downloads the weights the first time a name is used; `list_aliases()` lists the supported names. `ReIDModel.from_pretrained()` with no name loads `osnet_x1_0_msmt17_combineall`.
2. **Saved model folder.** `ReIDModel.from_pretrained("/path/to/export")`, `ReIDModel.from_pretrained("hf://org/repo")`, or `ReIDModel.from_pretrained("https://example.com/models/my_model/")`. The folder must contain `weights.safetensors` and `reid_config.json`, with these names, as `save_pretrained` writes them.
3. **Weights file.** `ReIDModel.from_pretrained("weights.pth", architecture="osnet_x1_0")`. Hugging Face (`hf://org/repo/file.pth`), Google Drive (`gd://<file-id>`), and web (`https://.../weights.pth`) sources work the same way. The one difference: a `.pth` from the web must load with PyTorch's safe `weights_only=True`, because a checkpoint that needs full unpickling can run code. Convert such a file to `.safetensors`, or download it and pass the local path if you trust its source. Name the architecture, because the file carries no model configuration; `list_architectures()` lists the options. If most of the architecture's weights are missing from the file, `from_pretrained` raises an error, because the weights come from a different model. A partial load above that, such as a checkpoint with a different head, loads with a warning, and the missing parameters keep random values. So make sure the weights file was trained with the architecture you name.
4. **Random initialization.** `ReIDModel.from_pretrained(architecture="osnet_x1_0")`. The weights are random, so the embeddings carry no identity information; use this only to test the API without downloading weights.

```python
from reid import ReIDModel

model = ReIDModel.from_pretrained("osnet_x1_0_msmt17_combineall")
model = ReIDModel.from_pretrained("/path/to/export")
model = ReIDModel.from_pretrained("https://example.com/models/my_model/")
model = ReIDModel.from_pretrained("weights.pth", architecture="osnet_x1_0")
```

`model.save_pretrained(directory)` writes the weights to `weights.safetensors` and the architecture and preprocessing settings to `reid_config.json`, so `ReIDModel.from_pretrained(directory)` restores the same model.

To share a model you trained, save it with `save_pretrained`, upload the folder, and load it by its URL. `from_pretrained` looks for these two files in the folder, under these names:

| File                  | Format      | Contents                                                                                       |
| :-------------------- | :---------- | :--------------------------------------------------------------------------------------------- |
| `weights.safetensors` | safetensors | the model weights                                                                              |
| `reid_config.json`    | JSON        | `architecture`, the name `list_architectures()` lists, and `preprocessing`, the input settings |

Prefer this to sharing the weights file alone: `reid_config.json` sets the architecture and preprocessing, so nobody has to name them, and every weight in the file must match that architecture, where a bare weights file only needs half of them to.

## Pretrained models

| Name                                     | Architecture             | Trained on                          | Weights from                                                                                                          |
| :--------------------------------------- | :----------------------- | :---------------------------------- | :-------------------------------------------------------------------------------------------------------------------- |
| `osnet_x1_0_msmt17_combineall` (default) | OSNet x1.0               | MSMT17, all splits combined         | [torchreid model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO), via Hugging Face `kaiyangzhou/osnet` |
| `fastreid_mot17_sbs50`                   | FastReID SBS, ResNeSt-50 | MOT17 train-half ground-truth crops | [BoT-SORT](https://github.com/NirAharon/BoT-SORT) release, via Google Drive                                           |
| `osnet_x1_0_mot17`                       | OSNet x1.0               | MOT17 train-half crops              | Roboflow, fine-tuned for [Trackers](https://github.com/roboflow/trackers)                                             |
| `osnet_x1_0_sportsmot`                   | OSNet x1.0               | SportsMOT train crops               | Roboflow, fine-tuned for [Trackers](https://github.com/roboflow/trackers)                                             |
| `osnet_x1_0_dancetrack`                  | OSNet x1.0               | DanceTrack train crops              | Roboflow, fine-tuned for [Trackers](https://github.com/roboflow/trackers)                                             |
| `osnet_x1_0_soccernet`                   | OSNet x1.0               | SoccerNet train crops               | Roboflow, fine-tuned for [Trackers](https://github.com/roboflow/trackers)                                             |

Weights download the first time a model is loaded and are kept on disk, so later loads skip the download. See [weight cache](#weight-cache) for more details.

## Preprocessing

Each architecture has default preprocessing: OSNet resizes crops to 256 × 128 and FastReID SBS to 384 × 128, both by stretching. Pass a `ReIDPreprocessing` to `from_pretrained(..., preprocessing=...)` to override it. These are its fields, and what `reid_config.json` stores under `"preprocessing"`:

| Field           | Meaning                                                                               | Typical                                   |
| :-------------- | :------------------------------------------------------------------------------------ | :---------------------------------------- |
| `input_size`    | Image height and width                                                                | OSNet `(256, 128)`; FastReID `(384, 128)` |
| `resize_mode`   | `stretch` or `letterbox`                                                              | `stretch`                                 |
| `interpolation` | Image resizing method                                                                 | `bilinear`                                |
| `to_rgb`        | Swap BGR to RGB for crops cut from video frames; crops read from disk are already RGB | `True`                                    |
| `mean` / `std`  | Per-channel normalization                                                             | ImageNet values                           |
| `pad_value`     | Padding pixel value for `letterbox`                                                   | `114`                                     |

`extract_embeddings` returns raw embedding vectors; evaluation normalizes them when it computes cosine distance.

## Weight cache

Google Drive (`gd://...`) downloads are cached under `~/.cache/reid/weights/` by file name, and web (`https://...`) downloads by their host and path, for example `~/.cache/reid/weights/example.com/models/my_model/weights.safetensors`. Set `REID_CACHE_DIR` to move the cache; weights then go to `$REID_CACHE_DIR/weights/`. Hugging Face weights use the standard Hugging Face Hub cache.
