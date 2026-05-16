"""Convert trained Keras models to TensorFlow.js layers format.

Output structure:
    docs/models/cnn_custom/model.json  + weight shards
    docs/models/mobilenet_ft/model.json + weight shards

Run from the project root:
    python scripts/export_tfjs.py
"""
from pathlib import Path

import tensorflowjs as tfjs
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_MODELS_DIR = PROJECT_ROOT / "docs" / "models"

EXPORTS = [
    {
        "src": MODELS_DIR / "cnn_custom.keras",
        "fallback": MODELS_DIR / "cnn_custom.h5",
        "dst": DOCS_MODELS_DIR / "cnn_custom",
        "name": "cnn_custom",
    },
    {
        "src": MODELS_DIR / "mobilenet_ft.keras",
        "fallback": MODELS_DIR / "mobilenet_ft.h5",
        "dst": DOCS_MODELS_DIR / "mobilenet_ft",
        "name": "mobilenet_ft",
    },
]


def _resolve_path(entry: dict) -> Path:
    if entry["src"].exists():
        return entry["src"]
    if entry["fallback"].exists():
        return entry["fallback"]
    raise FileNotFoundError(
        f"Neither {entry['src']} nor {entry['fallback']} found. "
        "Train the models before exporting."
    )


def export_model(entry: dict) -> None:
    src = _resolve_path(entry)
    dst = entry["dst"]
    dst.mkdir(parents=True, exist_ok=True)

    print(f"\n[{entry['name']}] loading {src} ...")
    model = tf.keras.models.load_model(str(src))

    in_shape = model.input_shape
    out_shape = model.output_shape
    print(f"  input shape : {in_shape}")
    print(f"  output shape: {out_shape}")

    print(f"  converting → {dst} ...")
    tfjs.converters.save_keras_model(model, str(dst))

    shard_files = sorted(dst.glob("*.bin"))
    total_mb = sum(f.stat().st_size for f in shard_files) / 1e6
    print(f"  done — {len(shard_files)} weight shard(s), {total_mb:.1f} MB")


def main() -> None:
    print("TensorFlow.js model export")
    print(f"  TF version : {tf.__version__}")
    print(f"  tfjs version: {tfjs.__version__}")

    for entry in EXPORTS:
        export_model(entry)

    print("\nAll models exported successfully.")
    print(f"Files are in: {DOCS_MODELS_DIR}")


if __name__ == "__main__":
    main()
