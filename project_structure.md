# Project structure

Authoritative folder layout for the project. `CLAUDE.md` defers all
structure decisions to this file. Phases must respect this tree.

## Tree

```
sales-receptivity-cnn/
├── CLAUDE.md                  # Context and rules for Claude Code
├── README.md                  # Project overview (phase index until phase 9)
├── project_structure.md       # This file
├── requirements.txt           # Pinned dependencies (TF 2.10 + GPU on Windows)
├── .gitignore
├── phases/                    # Phase-by-phase implementation plan
│   ├── phase_0_setup.md
│   ├── phase_1_base_modules.md
│   ├── phase_2_notebook_eda.md
│   ├── phase_3_notebook_preprocessing.md
│   ├── phase_4_inference_modules.md
│   ├── phase_5_notebook_training.md
│   ├── phase_6_notebook_evaluation.md
│   ├── phase_7_api.md
│   ├── phase_8_demo.md
│   └── phase_9_polish.md
├── notebooks/                 # Pedagogical notebooks (phases 2, 3, 5, 6)
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
├── src/                       # Reusable Python modules (importable, no side effects)
│   ├── __init__.py
│   ├── config.py              # Paths, labels, mappings, hyperparameters
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # FER2013 loading + RGB/resize helpers
│   │   └── augmentation.py    # ImageDataGenerator pipelines
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_custom.py      # Custom 4-block CNN builder
│   │   ├── mobilenet_finetune.py  # MobileNetV2 head + unfreeze utility
│   │   └── trainer.py         # Callbacks + train_model + save_history
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── face_detector.py       # OpenCV Haar Cascade wrapper
│   │   ├── emotion_classifier.py  # Keras model wrapper for prediction
│   │   └── receptivity_mapper.py  # Emotion → score/signal + sliding index
│   └── api/
│       ├── __init__.py
│       ├── main.py            # FastAPI app and endpoints
│       └── schemas.py         # Pydantic request/response models
├── demo/                      # Streamlit demo
│   └── app.py
├── models/                    # Trained model artifacts (gitignored)
│   ├── cnn_custom.keras
│   ├── mobilenet_ft.keras
│   └── histories/             # Tracked in git: needed for plot regeneration
│       ├── cnn_custom_history.json
│       └── mobilenet_ft_history.json
├── data/                      # Datasets (gitignored)
│   ├── raw/                   # FER2013 train/<emotion>/*.jpg, test/<emotion>/*.jpg
│   └── processed/             # fer2013_gray.npz, fer2013_rgb64.npz, class_weights.json
└── tests/
    ├── test_inference.py      # FaceDetector + ReceptivityIndex + EmotionClassifier
    └── test_api.py            # /health and /predict/image endpoints
```

## Rules

- `src/` is importable without side effects: no training, no data loading,
  no model loading at import time
- Notebooks consume `src/` — they never re-implement logic that belongs in `src/`
- Saved model weights go in `models/` in `.keras` format (fallback to `.h5` if
  `.keras` fails on TF 2.10)
- Training histories go in `models/histories/` as JSON and ARE tracked in git
- Raw data lives in `data/raw/` (gitignored). Processed tensors live in
  `data/processed/` (gitignored)
- Notebooks are numbered `01_`, `02_`, `03_`, `04_` following the phase order
