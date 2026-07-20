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
```

## Ways to load a model

Use `ReIDModel.from_pretrained(...)`. Pick the form that matches what you have:

1. **Default / curated alias** — `ReIDModel.from_pretrained()` or
   `ReIDModel.from_pretrained("osnet_x1_0_msmt17_combineall")`.
   Use when you want a known library recipe. The alias resolves through the
   registry to architecture, weights URL, and preprocessing. Other shipped
   aliases include `fastreid_mot17_sbs50`.

2. **Saved directory or HF repo with `reid_config.json`** —
   `ReIDModel.from_pretrained("/path/to/export")` or
   `ReIDModel.from_pretrained("hf://org/repo")`.
   Use after `save_pretrained` (or an equivalent Hub upload). The config names
   the architecture and preprocessing; weights come from
   `weights.safetensors` next to the config.

3. **Bare checkpoint file** —
   `ReIDModel.from_pretrained("weights.pth", architecture="osnet_x1_0")`
   (also works for `hf://.../file.pth` and `gd://...`).
   Use when you only have weights and already know the backbone. Architecture
   is required; preprocessing defaults from the architecture unless you pass
   `preprocessing=`.

4. **Architecture only (random init)** —
   `ReIDModel.from_pretrained(architecture="osnet_x1_0")` with no weights
   source.
   Use for tests, scaffolding, or before you train / attach a checkpoint.

## Package layout

The public API is exported from the top-level `reid` package; advanced helpers
live in submodules:

```
src/reid/
├── __init__.py         # narrow public surface (see __all__)
├── model.py            # ReIDModel (the encoder)
├── encoder.py          # ReIDEncoder protocol
├── preprocessing.py    # ReIDPreprocessing (crop + tensor preprocessing)
├── catalog.py          # curated alias catalog: ModelCard, ALIASES, resolve_model_card
├── architectures/      # architecture dispatch: build_architecture, list_architectures
├── loaders/            # checkpoint I/O: weight resolution, state-dict loading, load plans
│   ├── __init__.py     #   re-exports the public loader helpers
│   └── io.py           #   implementation
└── eval/               # gallery evaluation: datasets, metrics, evaluator
```

There are **two distinct registries**, kept separate on purpose:

- **Architecture dispatch** lives in `architectures/` — `build_architecture` /
  `list_architectures` map an architecture name (`osnet_x1_0`, `timm:<name>`,
  `fastreid_sbs_resnest50`) to an `nn.Module`.
- **Curated alias catalog** lives in top-level `catalog.py` — `ALIASES` maps a
  friendly alias (`osnet_x1_0_msmt17_combineall`, `fastreid_mot17_sbs50`) to a
  `ModelCard` bundling architecture + weights URL + preprocessing.

## Adding a new architecture

1. Implement the module under `src/reid/architectures/` (or use `timm:<name>`).
2. Register it in `src/reid/architectures/__init__.py` via `build_architecture` /
   `list_architectures`.
3. If bare `.pth` loads need non-default crop behaviour, add an entry in
   `ARCHITECTURE_DEFAULT_PREPROCESSING` in `src/reid/catalog.py`.
4. Optionally add a curated alias in `ALIASES` in the same catalog file.

## Gallery eval

```python
from reid import ReIDEvaluator, ReIDModel, load_market1501

model = ReIDModel.from_pretrained()
query, gallery = load_market1501("/path/to/Market-1501")
result = ReIDEvaluator(model).evaluate(query, gallery)
print(result.metrics)
```

To reproduce torchreid model-zoo OSNet numbers on Market-1501 and MSMT17 (same-domain checkpoints, cosine vs Euclidean), run [`notebooks/eval_reid.ipynb`](notebooks/eval_reid.ipynb) — Colab-friendly, pins this PR branch while under review.

## Weight cache

Google Drive downloads (`gd://...`) are cached under `~/.cache/reid/weights/`. Set the `REID_CACHE_DIR` environment variable to override the cache root (weights are then stored under `$REID_CACHE_DIR/weights/`). Hugging Face weights use the standard Hugging Face Hub cache.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
