# Phase 0 — Initial setup

## Objective

Project skeleton ready to start coding. No business logic yet.

## Tasks

- Create folder structure:
  ```
  project/
  ├── notebooks/
  ├── src/
  │   ├── data/
  │   ├── models/
  │   ├── inference/
  │   └── api/
  ├── demo/
  ├── models/
  │   └── histories/
  ├── data/
  │   ├── raw/
  │   └── processed/
  └── tests/
  ```
- Add `__init__.py` files where needed inside `src/`
- Create Python virtual environment and install `requirements.txt`
- Verify GPU detection with a small script:
  ```python
  import tensorflow as tf
  print("TF version:", tf.__version__)
  print("GPUs:", tf.config.list_physical_devices('GPU'))
  ```
- Configure Kaggle API:
  - Place `kaggle.json` in `%USERPROFILE%\.kaggle\kaggle.json` on Windows
  - Verify with `kaggle datasets list -s fer2013`
- Download FER2013 to `data/raw/`:
  - `kaggle datasets download -d msambare/fer2013 -p data/raw/ --unzip`
- Create `.gitignore` with:
  ```
  data/
  models/*.keras
  models/*.h5
  .ipynb_checkpoints/
  __pycache__/
  *.pyc
  venv/
  .env
  .kaggle/
  ```
  Note: `models/histories/*.json` is intentionally NOT ignored — Notebook 4
  regenerates training plots from these files without retraining.
- Create `src/config.py` with:
  - Path constants (`PROJECT_ROOT`, `DATA_DIR`, `MODELS_DIR`, etc.) using `pathlib.Path`
  - `EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']`
  - `EMOTION_TO_SCORE` dict (the mapping from the case study)
  - `EMOTION_TO_SIGNAL` dict with the commercial signal text
  - Default hyperparameters: `IMG_SIZE_CUSTOM = (48, 48)`, `IMG_SIZE_MOBILENET = (64, 64)`,
    `BATCH_SIZE = 64`, `RANDOM_SEED = 42`
- Initialize git repo

## Validation

- `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
  prints at least one GPU device
- FER2013 train/test folders exist under `data/raw/`
- `from src import config` works from any notebook

## Notes

- If GPU is not detected, install CUDA 11.2 and cuDNN 8.1 (specific versions for TF 2.10).
  This is a one-time setup.
- Do not proceed to Phase 1 until GPU is verified or you have decided to fall back to CPU.

## Output

Empty but runnable project. Next phase: build reusable modules in `src/`.
