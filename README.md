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

## Adding a new architecture

1. Implement the module under `src/reid/architectures/` (or use `timm:<name>`).
2. Register it in `src/reid/architectures/__init__.py` via `build_architecture` /
   `list_architectures`.
3. If bare `.pth` loads need non-default crop behaviour, add an entry in
   `ARCHITECTURE_DEFAULT_PREPROCESSING` in `src/reid/models/registry.py`.
4. Optionally add a curated alias in `ALIASES` in the same registry file.

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

Google Drive downloads (`gd://...`) are cached under `~/.cache/trackers/weights/` for behavioral parity with Trackers. Hugging Face weights use the standard Hugging Face Hub cache.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
