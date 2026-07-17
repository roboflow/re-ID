# re-ID

Re-identification model stack for extracting appearance embeddings from detections and image crops. This package ports the model loading, preprocessing, and inference stack from [Roboflow Trackers](https://github.com/roboflow/trackers) as a standalone library.

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

Curated aliases such as `osnet_x1_0_msmt17_combineall` and `fastreid_mot17_sbs50` are also supported via `ReIDModel.from_pretrained(source=...)`.

## Weight cache

Google Drive downloads (`gd://...`) are cached under `~/.cache/trackers/weights/` for behavioral parity with Trackers. Hugging Face weights use the standard Hugging Face Hub cache.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
