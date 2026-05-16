# Sales Receptivity CNN

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-6366f1?style=for-the-badge&logo=github)](https://jorgepulgar.github.io/sales-receptivity-cnn/)

**[▶ Open live demo](https://jorgepulgar.github.io/sales-receptivity-cnn/)** — runs entirely in the browser, no install required.

![Web demo — neutral face, 63 % confidence](docs/neutral63.png)

> **Note:** The web demo uses BlazeFace for face detection while the local Streamlit
> demo uses OpenCV Haar Cascades. Bounding boxes differ between the two detectors,
> so predictions are not bit-identical across demos — this is expected behaviour.

A real-time emotional-receptivity analyser for sales presentations. The
system reads a video stream (recorded file or live webcam), detects the
prospect's face, classifies their emotional state with a CNN trained on
FER2013, and produces a continuous **receptivity index** (0–10) that the
salesperson can use as a coaching signal — peaks above 7 suggest the
prospect is engaged, valleys below 3 suggest disengagement or discomfort.

The project compares two architectures end-to-end: a custom four-block
CNN trained from scratch on the native 48×48 grayscale FER2013 input,
and a MobileNetV2 fine-tuned from ImageNet weights on 96×96 RGB inputs.
Both are exposed through a FastAPI service and a Streamlit demo that
share the same `src/inference/` modules — no logic is duplicated.

This is a Deep Learning class project. Reproducibility, didactic value
and a clean record of decisions take priority over extreme robustness.

## Headline results

| Model | Test accuracy | F1 macro | Parameters | CPU inference (ms / frame) |
|---|---|---|---|---|
| Custom CNN (4-block) | 60.1 % | 0.499 | 1.70 M | ~27 |
| **MobileNetV2 (fine-tuned)** | **63.0 %** | **0.601** | **2.59 M** | **~29** |

MobileNetV2 wins the multi-criteria scoring in Notebook 4 (weighted
total **0.70 vs 0.30** for the custom CNN) and is the deployed model
for the API and Streamlit demo. The 10-point gap in F1 macro is driven
almost entirely by the rare **Disgust** class: the custom CNN literally
never predicts Disgust (F1 = 0.00), while MobileNetV2 reaches F1 = 0.52
once its input resolution is large enough for the backbone to retain
useful spatial features (see *Methodology highlights* below).

Real-world accuracy on live webcam frames is typically 5–10 points
lower than the FER2013 test set due to lighting, pose, and motion
blur differences. Treat ~55 % as the realistic ceiling in the demo.

## Architecture

```
                    ┌─────────────────────────────────┐
                    │   Video source                  │
                    │   - recorded .mp4 / .avi / .mov │
                    │   - live webcam (cv2.VideoCapture)│
                    └────────────────┬────────────────┘
                                     │ BGR frame
                                     ▼
                    ┌─────────────────────────────────┐
                    │   FaceDetector                  │
                    │   OpenCV Haar Cascade           │
                    │   returns largest bbox          │
                    └────────────────┬────────────────┘
                                     │ ROI
                                     ▼
                    ┌─────────────────────────────────┐
                    │   EmotionClassifier             │
                    │   MobileNetV2 (96×96×3, RGB)    │
                    │   or custom CNN (48×48×1, gray) │
                    │   returns (emotion, confidence) │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │   ReceptivityIndex              │
                    │   rolling weighted average      │
                    │   over a sliding window         │
                    │   returns 0–10 score            │
                    └────────────────┬────────────────┘
                                     │
                  ┌──────────────────┴──────────────────┐
                  ▼                                     ▼
         ┌──────────────────┐                ┌──────────────────┐
         │   FastAPI        │                │   Streamlit demo │
         │   /predict       │                │   recorded video │
         │   /predict/batch │                │   live webcam    │
         │   /predict/video │                │                  │
         └──────────────────┘                └──────────────────┘
```

The receptivity mapping uses a hand-crafted heuristic
(`src/inference/receptivity_mapper.py`): happy = 9, surprise = 7,
neutral = 5, sad = 3, fear = 2, angry/disgust = 1. The sliding window
defaults to 10 frames and predictions are weighted by softmax
confidence — a noisy low-confidence prediction barely moves the index.

## Stack

- **Python 3.10**
- **TensorFlow 2.10.1** (last version with native CUDA GPU support on
  Windows; locks the version of several transitive deps — see `dev_log.md`)
- **FastAPI + uvicorn** for the inference service
- **Streamlit 1.18.1** for the demo (pinned because >= 1.19 requires
  protobuf >= 3.20 which is incompatible with TF 2.10)
- **OpenCV 4.9** for face detection and webcam capture
- **FER2013** dataset (35 887 grayscale 48×48 face images, 7 emotion classes)

## Installation

### Prerequisites

- Windows 10/11 or Linux. Native CUDA GPU support requires Windows
  with CUDA 11.2 + cuDNN 8.1 (only for training; the demo runs fine
  on CPU at ~30 ms/frame).
- [Miniconda](https://docs.conda.io/projects/miniconda/) installed.

### Environment setup

```bash
conda create -n sales-cnn python=3.10 -y
conda activate sales-cnn
pip install -r requirements.txt
```

If you have an NVIDIA GPU and want to train, install the CUDA Toolkit
11.2 and cuDNN 8.1 separately. Verify with:

```python
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
```

If you see a `PhysicalDevice` line, GPU training is ready. Otherwise
training will fall back to CPU (much slower).

## Dataset setup

The FER2013 dataset is downloaded from Kaggle. Configure the Kaggle API
once with your `~/.kaggle/kaggle.json` token, then:

```bash
kaggle datasets download -d msambare/fer2013 -p data/raw/ --unzip
```

After this `data/raw/train/<emotion>/*.jpg` and
`data/raw/test/<emotion>/*.jpg` should exist (~60 MB raw, ~35 887 images).

`data/raw/` is gitignored — every collaborator must re-download.

## Running the project

### 1. Notebooks

Run the four notebooks in order; each consumes the previous one's output.
All notebooks restart cleanly with `Kernel → Restart & Run All`.

| # | Notebook | What it does | Approx time (GPU) |
|---|---|---|---|
| 01 | `notebooks/01_eda.ipynb` | Class-distribution analysis, sample grids, per-class mean faces, cosine similarity matrix | < 1 min |
| 02 | `notebooks/02_preprocessing.ipynb` | Normalisation, 96×96 RGB resize, one-hot labels, class weights JSON, augmentation pipeline. Writes `data/processed/fer2013_gray.npz` (~107 MB), `fer2013_rgb96.npz` (~991 MB), `class_weights.json` | 1–2 min |
| 03 | `notebooks/03_model_training.ipynb` | Trains the custom CNN (75 epochs) and MobileNetV2 (Stage 1: 20 epochs head-only, Stage 2: 30 epochs with top 60 backbone layers unfrozen). Saves `models/cnn_custom.keras`, `models/mobilenet_ft.keras` and the history JSONs | 8–25 min |
| 04 | `notebooks/04_evaluation.ipynb` | Loads both saved models, computes test accuracy / F1 / latency, plots training curves and confusion matrices, runs Grad-CAM on the winning model, simulates a 200-frame sales call with the receptivity index, picks the deployment model via weighted scoring | 2–4 min |

### 2. FastAPI service

The API exposes the inference pipeline as REST endpoints. It auto-selects
the best available model from `models/` (prefers MobileNetV2 over the
custom CNN, prefers `.keras` over `.h5`).

```bash
uvicorn src.api.main:app --reload
```

Swagger UI: <http://localhost:8000/docs>

Endpoints:

- `GET  /health` — service status and which model is loaded
- `POST /predict/image` — single JPEG/PNG → emotion + confidence + receptivity score
- `POST /predict/batch` — list of frames, returns one prediction per frame
- `POST /predict/video` — video file, returns per-frame analyses + session summary

### 3. Streamlit demo

```bash
streamlit run demo/app.py
```

The browser opens at <http://localhost:8501>. Two modes in the sidebar:

- **Recorded video**: upload `.mp4 / .avi / .mov`. The app samples one
  frame per second, produces a receptivity timeline, a per-emotion pie
  chart, key-moment frame grids and a downloadable CSV.
- **Webcam**: tick **▶ Run live analysis** to start continuous capture
  (~8 FPS). Each frame: face detect → emotion predict → receptivity
  index update → bbox + label overlay drawn on the displayed frame.
  Untick to stop and release the camera. Session state (rolling
  receptivity, emotion history) persists across pauses; the **Reset
  session** button clears it.

The demo imports `src.inference` directly rather than going through the
FastAPI endpoint — running two processes during a live presentation
adds failure points without any benefit on the same machine.

### 4. Tests

```bash
pytest tests/ -v
```

Tests cover the `FaceDetector`, `ReceptivityIndex`, `EmotionClassifier`
unit behaviour and the `/health` / `/predict/image` endpoints. Tests
that require a trained model are skipped if `models/` is empty.

## Methodology highlights

The project hit several non-obvious failure modes during training. All
of them are documented in detail in `dev_log.md`; this is the short
version of what made the final results possible.

### TF 2.10 weighted-loss bug

Passing `class_weight` or `sample_weight` to `model.fit()` on TF 2.10
with one-hot labels collapses training to uniform output: loss freezes
at `log(num_classes)` and val_accuracy stays at chance level for the
entire run. Both routes are broken — `class_weight` directly, and
`sample_weight` through a `keras.utils.Sequence` returning `(x, y, w)`
triples.

The workaround is to **not use weighting at all** and absorb the FER2013
imbalance with augmentation + light dropout. The class-weight values are
still computed and saved to `data/processed/class_weights.json` as a
documentation artefact.

### Dropout vs augmentation

The first training attempts with the conventional 0.25 / 0.5 dropout
values combined with online augmentation **locked the optimiser into
predicting Happy for every input** — val_accuracy froze at exactly
0.2512 (Happy's share of the validation set) for all 50 epochs.
Reducing dropout to 0.1 (conv) / 0.3 (dense) restored learning: the
total regularisation noise from augmentation plus dropout was simply
too high for the optimiser to find any direction better than the
trivial majority-class minimum.

### MobileNetV2 input resolution

The first MobileNetV2 attempt used 64×64 inputs to keep inference
latency low. This left the backbone with a 2×2 spatial feature map
after its ~32× internal downsampling — almost no information for
`GlobalAveragePooling2D` to summarise. Stage 2 stalled at 45 % val
accuracy. Raising the input to **96×96** (the official minimum useful
size for MobileNetV2) gave the backbone a 3×3 feature map and recovered
the entire 15-point gap, putting MobileNetV2 ahead of the custom CNN.

### Label smoothing

Both losses use `categorical_crossentropy(label_smoothing=0.1)`. FER2013
contains genuinely ambiguous expressions (Sad / Neutral, Angry / Disgust)
where a one-hot target is incorrect even for a perfect classifier.
Smoothing softens those targets and typically adds 0.5–1 pt of val
accuracy at zero cost.

## Project structure

```
sales-receptivity-cnn/
├── CLAUDE.md                       # Context and rules for Claude Code
├── README.md                       # This file
├── dev_log.md                      # Detailed bug / decision history
├── project_structure.md            # Authoritative folder layout reference
├── requirements.txt                # Pinned dependencies
├── .gitignore
├── phases/                         # Phase-by-phase implementation plan
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_preprocessing.ipynb      # Normalisation, resize, save .npz
│   ├── 03_model_training.ipynb     # Train both models, save artefacts
│   └── 04_evaluation.ipynb         # Quantitative + qualitative evaluation
├── src/                            # Reusable modules; importable without side effects
│   ├── config.py                   # Paths, labels, hyperparameters
│   ├── data/
│   │   ├── loader.py               # FER2013 loading + RGB/resize helpers
│   │   └── augmentation.py         # cv2-based augmentation pipeline
│   ├── models/
│   │   ├── cnn_custom.py           # 4-block CNN builder
│   │   ├── mobilenet_finetune.py   # MobileNetV2 head + unfreeze utility
│   │   └── trainer.py              # Callbacks + train_model + save_history
│   ├── inference/
│   │   ├── face_detector.py        # OpenCV Haar Cascade wrapper
│   │   ├── emotion_classifier.py   # Keras model wrapper
│   │   └── receptivity_mapper.py   # Emotion → score/signal + sliding index
│   └── api/
│       ├── main.py                 # FastAPI app
│       └── schemas.py              # Pydantic request / response models
├── demo/
│   └── app.py                      # Streamlit two-mode demo
├── models/                         # Trained artefacts (.keras gitignored)
│   └── histories/                  # Training histories (tracked in git)
├── data/                           # Datasets (gitignored)
│   ├── raw/                        # Kaggle FER2013 download target
│   └── processed/                  # Generated .npz + class_weights.json
└── tests/
    ├── test_inference.py
    └── test_api.py
```

## Limitations and ethical considerations

### Generalisation from FER2013 to real webcam conditions

FER2013 images are 48×48 crops collected from internet searches under
controlled labelling conditions. A live webcam stream introduces motion
blur, variable lighting, partial occlusion, and head-pose variation
that FER2013 under-represents. The ~63 % test accuracy reported here
is an **upper bound** on field performance; informal experiments with
the webcam mode show 50–55 % is more realistic.

### Limits of the emotion → receptivity heuristic

The mapping from emotion to receptivity score is a hand-crafted
heuristic, **not an empirically validated model of sales psychology**.
A prospect who is laughing politely while concealing scepticism
produces a high index — a false positive. A highly engaged analyst
asking fear-coded probing questions produces a low index — a false
negative. The system should be positioned as a coaching aid providing
a second opinion, not as a ground-truth signal.

### Demographic bias

FER2013 is biased toward Western, lighter-skinned faces. Models trained
on it have documented higher error rates for darker skin tones and for
non-Western facial-expression norms. Deploying this system in diverse
sales teams or cross-cultural settings without explicit bias auditing
risks systematically misreading certain populations, with downstream
consequences for how their performance is judged.

### Is 63 % accuracy acceptable?

For a coaching tool that aggregates over a 10-frame rolling window,
individual misclassifications matter less than persistent patterns: a
sustained 8-frame fear signal is meaningful even if 2 of those frames
were misclassified as sad. For **deterministic decisions** —
automatically flagging a call as failed, or penalising a salesperson —
63 % accuracy is far too low. The distinction between coaching aid
and decision tool must be communicated to every end user.

### What this version does *not* fix

- **Disgust recall** is only 37 % (MobileNetV2). The 96×96 upgrade made
  the class learnable at all, but FER2013's 1.5 % Disgust support is
  genuinely too thin without minority-class oversampling or AffectNet
  pretraining.
- **Fear ↔ Surprise** confusion persists. Both classes share wide-eye
  geometry and FER2013 has visibly mislabelled samples in this pair.
- **MobileNetV2 train/val gap** reaches ~12 pts by the end of Stage 2.
  Shorter Stage-2 training or stronger augmentation would close it at
  the cost of 0.5–1 pt of best val accuracy.

## Future improvements

These are documented but not implemented:

- **Multimodal fusion** — combining facial expression with voice tone
  (pitch, pace, energy) substantially reduces single-modality errors.
- **AffectNet pretraining** — a backbone pre-trained on AffectNet
  (450 k+ diverse labelled faces) before FER2013 fine-tuning would
  improve both accuracy and demographic fairness.
- **TFLite / ONNX deployment** — quantised mobile-format conversion
  would push inference under 5 ms on-device.
- **Fine-tuning on real sales-call frames** — even a small labelled
  set would correct the domain shift that FER2013 cannot.
- **Cheap CNN-side wins** — test-time augmentation and a 3-seed
  ensemble can each add 1–2 pts without architectural changes.

## License

MIT. See `LICENSE` if/when added — this is a student project intended
to be shared with classmates and instructors.

## Acknowledgements

- FER2013 dataset by Pierre-Luc Carrier and Aaron Courville, made
  available via the [Kaggle Facial Expression Recognition Challenge](https://www.kaggle.com/datasets/msambare/fer2013).
- MobileNetV2 from Sandler et al., 2018, distributed by Keras
  Applications under the Apache 2.0 license.
