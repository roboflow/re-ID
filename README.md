# re-ID

Re-identification models and gallery evaluation for appearance embeddings. This package ports the model loading, preprocessing, inference, and Market/MSMT evaluation stack from [Roboflow Trackers](https://github.com/roboflow/trackers) as a standalone library.

## Install

```bash
pip install -e ".[dev]"
```

Or with uv:

```bash
uv sync --all-extras
```

Once published, install from PyPI with `pip install reid`; import as `reid`.

## Quick start

```python
import numpy as np
import supervision as sv
from reid import ReIDModel

model = ReIDModel.from_pretrained(architecture="osnet_x0_25", device="cpu")

frame = np.zeros((256, 128, 3), dtype=np.uint8)
detections = sv.Detections(xyxy=np.array([[0, 0, 128, 256]], dtype=np.float32))
embeddings = model.extract_features(detections, frame)
print(embeddings.shape)  # (1, 512)

# Pre-cropped images on disk (gallery eval, etc.)
embeddings = model.extract_features_from_paths(["/path/to/crop.jpg"])
```

## Ways to load a model

Use `ReIDModel.from_pretrained(...)`. Pick the form that matches what you have:

1. **Curated alias** (or no argument): `ReIDModel.from_pretrained()` or
   `ReIDModel.from_pretrained("osnet_x1_0_msmt17_combineall")`.
   Resolves architecture and weights URL; preprocessing comes from that
   architecture's `default_preprocessing()`. Other shipped aliases include
   `fastreid_mot17_sbs50`.

2. **Saved directory or HF repo with `reid_config.json`**:
   `ReIDModel.from_pretrained("/path/to/export")` or
   `ReIDModel.from_pretrained("hf://org/repo")`.
   Use after `save_pretrained`. The config records architecture and
   preprocessing; weights live in `weights.safetensors` next to it.

3. **Bare checkpoint file** plus `architecture=`:
   `ReIDModel.from_pretrained("weights.pth", architecture="osnet_x1_0")`
   (also `hf://.../file.pth` and `gd://...`).
   Preprocessing comes from that architecture's `default_preprocessing()`
   unless you pass `preprocessing=`.

4. **Architecture only** (random init):
   `ReIDModel.from_pretrained(architecture="osnet_x1_0")`.
   Useful for tests, scaffolding, or before training attaches weights.

### Preprocessing and config

Each architecture class owns its default crop/tensor recipe as a
`default_preprocessing()` classmethod (`OSNet.default_preprocessing()`,
`FastReIDSBSResNeSt50.default_preprocessing()`). OSNet uses 256x128 stretch;
FastReID SBS uses 384x128 stretch (BoT-SORT FastReIDInterface). String dispatch
is `architectures.default_preprocessing_for_architecture`.

`ReIDPreprocessing` fields (also what `reid_config.json` stores under
`"preprocessing"`):

| Field | Meaning | Typical |
| --- | --- | --- |
| `input_size` | `(H, W)` | OSNet `(256, 128)`; FastReID `(384, 128)` |
| `resize_mode` | `stretch` or `letterbox` | `stretch` |
| `interpolation` | OpenCV resize | `bilinear` |
| `to_rgb` | BGR→RGB swap for video-frame crops only; on-disk RGB paths are unchanged | `True` |
| `mean` / `std` | tensor Normalize | ImageNet |
| `pad_value` | letterbox pad | `114` |

Embedding L2 norm is not a preprocessing field: extract returns raw vectors;
cosine distance normalizes inside `ReIDEvaluator`.

`save_pretrained` writes `architecture` and `preprocessing` into
`reid_config.json`, with weights beside it in `weights.safetensors`. Domain
warnings stay on curated aliases at load time; they are not written into saved
configs.

## Package layout

The public API is exported from the top-level `reid` package; advanced helpers
live in submodules. Discovery helpers: `list_aliases()`, `list_architectures()`.
Constants: `DEFAULT_MODEL`, `FASTREID_MOT17_SBS50`.

```
src/reid/
├── __init__.py         # narrow public surface (see __all__)
├── model.py            # ReIDModel (the encoder)
├── preprocessing.py    # ReIDPreprocessing (shared crop/tensor type)
├── catalog.py          # curated aliases (ModelCard, ALIASES)
├── architectures/      # builders + class default_preprocessing() on each net
├── loaders.py          # weight I/O and resolve_load_plan
├── data/               # ReIDSplit, benchmarks, identity crops, MOT patches
├── eval/               # gallery evaluation (metrics + ReIDEvaluator)
└── training/           # fine-tuning loop (datasets, losses, trainer)
```

- **Architectures** know how to build a net and expose preprocessing via
  `default_preprocessing()` on the architecture class.
- **Catalog** is only the small allowlist of named recipes (alias →
  architecture + weights URL + domain warning). Preprocessing for aliases is
  derived from the architecture at load time. Curated aliases require a 100%
  state-dict match at load time (enforced in `resolve_load_plan`).

## Adding a new architecture

1. Implement under `src/reid/architectures/` (or use `timm:<name>`).
2. Add a `@classmethod default_preprocessing()` on the architecture class,
   documenting the input size / resize / colour contract for that checkpoint family.
3. Register build + preprocessing dispatch in `architectures/registry.py`.
4. Optionally add a curated alias in `catalog.ALIASES`.

## Gallery eval

```python
from reid import ReIDEvaluator, ReIDModel, load_market1501

model = ReIDModel.from_pretrained()
query, gallery = load_market1501("/path/to/Market-1501")
result = ReIDEvaluator(model).evaluate(query, gallery)
print(result.metrics)
```

To reproduce torchreid model-zoo OSNet numbers on Market-1501 and MSMT17 (same-domain checkpoints, cosine vs Euclidean), run [`notebooks/eval_reid.ipynb`](notebooks/eval_reid.ipynb) — Colab-friendly, pins this PR branch while under review.

## Fine-tuning on MOT crops

The data layer follows a single flow: `source format -> loader -> ReIDSplit -> optional partitioning -> Dataset/evaluator`. Crop loading and partitioning live in `reid.data`; MOT crop generation lives in `reid.data.mot`. `reid.training` fine-tunes an encoder on a crop `ReIDSplit` and stays a submodule import so `import reid` does not pull the trainer.

`generate_mot_patches` reads MOT sequences and writes `output_dir/<seq>_<id>/<seq>_<frame>.jpg`. The default `split="train_half"` keeps crops frame-disjoint from the standard MOT17 val-half tracking evaluation, and MOT17 detector variants collapse to one base sequence.

```python
from reid.data.mot import generate_mot_patches
from reid.training import TrainConfig, train_reid

generate_mot_patches("/path/to/MOT17/train", "/path/to/crops", split="train_half")

result = train_reid(
    "/path/to/crops",
    TrainConfig(epochs=30),
    architecture="osnet_x1_0",
    pretrained="osnet_x1_0_msmt17_combineall",
    output_dir="/path/to/checkpoint",
)

from reid import ReIDModel

model = ReIDModel.from_pretrained("/path/to/checkpoint")
```

`train_reid` accepts either a crop dataset path or a `ReIDSplit`. A path is loaded with `reid.data.load_identity_crops`; pass a `ReIDSplit` directly when you have already loaded or partitioned the crops. It optimizes label-smoothed cross-entropy plus a batch-hard triplet term (optional center loss) with a PK sampler, and resolves architecture, warm-start weights, and preprocessing through the same `resolve_load_plan` entry point as `ReIDModel.from_pretrained`. Warm-start is a partial load: the fresh classifier head is not in the source checkpoint, so no strict state-dict match is enforced. When `output_dir` is set, the tuned encoder is written via `save_pretrained` and reloads with `ReIDModel.from_pretrained`.

To measure retrieval on held-out identities without cross-sequence leakage, `reid.data.build_identity_crop_holdout_split` takes a crop `ReIDSplit` and reserves a fraction of identities for a query/gallery split, returning the remaining crops as a train split. Query and gallery crops use synthetic camera ids 0 and 1 so single-camera MOT folders still produce valid retrieval pairs under gallery eval. See [`notebooks/train_reid_mot17.ipynb`](notebooks/train_reid_mot17.ipynb) for the MOT17 train-half fine-tune to BoT-SORT A/B walkthrough.

## Weight cache

Google Drive downloads (`gd://...`) are cached under `~/.cache/reid/weights/`. Set the `REID_CACHE_DIR` environment variable to override the cache root (weights are then stored under `$REID_CACHE_DIR/weights/`). Hugging Face weights use the standard Hugging Face Hub cache.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
