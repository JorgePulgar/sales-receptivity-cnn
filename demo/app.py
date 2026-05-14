# demo/app.py
# This module imports src.inference directly instead of routing through the
# FastAPI endpoint. Running two separate processes (uvicorn + streamlit) during
# a class presentation adds failure points and synchronisation overhead without
# any benefit: both processes share the same filesystem and the same GPU, so
# calling the endpoint would just be a local HTTP round-trip. Direct import
# keeps the demo self-contained and eliminates network errors during live demos.

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.inference.face_detector import FaceDetector
from src.inference.emotion_classifier import EmotionClassifier
from src.inference.receptivity_mapper import ReceptivityIndex, map_emotion_to_score


def _resolve_model():
    """Return (path, input_size, use_rgb) for the best available model, or Nones."""
    candidates = [
        (config.MODELS_DIR / "mobilenet_ft.keras", config.IMG_SIZE_MOBILENET, True),
        (config.MODELS_DIR / "mobilenet_ft.h5",    config.IMG_SIZE_MOBILENET, True),
        (config.MODELS_DIR / "cnn_custom.keras",   config.IMG_SIZE_CUSTOM, False),
        (config.MODELS_DIR / "cnn_custom.h5",      config.IMG_SIZE_CUSTOM, False),
    ]
    for p, size, rgb in candidates:
        if p.exists():
            return p, size, rgb
    return None, None, None

st.sidebar.title("Sales Receptivity CNN")
st.sidebar.markdown(
    "Emotion-based receptivity analyser for sales presentations. "
    "The CNN predicts the prospect's emotional state frame by frame and "
    "aggregates it into a rolling receptivity index."
)

mode = st.sidebar.selectbox("Mode", ["Recorded video", "Webcam"])
window_size = st.sidebar.slider(
    "Receptivity window size", min_value=5, max_value=30, value=10
)
weight_by_confidence = st.sidebar.checkbox("Weight by confidence", value=True)

_model_path, _input_size, _use_rgb = _resolve_model()
if _model_path is not None:
    st.sidebar.info(
        f"**Model loaded:** `{_model_path.name}`  \n"
        f"Input size: {_input_size[1]}×{_input_size[0]} px"
    )
else:
    st.sidebar.warning(
        "No trained model found in `models/`.  \n"
        "Train a model in Notebook 3 first."
    )

# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — Recorded video
# ══════════════════════════════════════════════════════════════════════════════

if mode == "Recorded video":
    st.title("Recorded Video Analysis")
    uploaded = st.file_uploader(
        "Upload a sales recording", type=["mp4", "avi", "mov"]
    )

    if uploaded is not None:
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)

        cap = cv2.VideoCapture(str(tmp_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        sample_step = max(1, int(fps))
        sample_indices = list(range(0, total_frames, sample_step))
        n_samples = len(sample_indices)

        if _model_path is None:
            cap.release()
            tmp_path.unlink(missing_ok=True)
            st.error("No model loaded — train one in Notebook 3.")
            st.stop()

        detector = FaceDetector()
        classifier = EmotionClassifier(_model_path, _input_size, _use_rgb)
        ri = ReceptivityIndex(window_size=window_size, weight_by_confidence=weight_by_confidence)

        progress_bar = st.progress(0, text="Analysing frames…")
        records: list[dict] = []
        face_misses = 0
        key_frames: dict[int, tuple] = {}

        for step, frame_idx in enumerate(sample_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                face_misses += 1
                progress_bar.progress((step + 1) / n_samples)
                continue
            bbox = detector.detect_largest(frame)
            if bbox is None:
                face_misses += 1
                progress_bar.progress((step + 1) / n_samples)
                continue
            roi = detector.extract_roi(
                frame, bbox, _input_size, to_grayscale=not _use_rgb
            )
            emotion, confidence, _ = classifier.predict(roi)
            idx_val = ri.update(emotion, confidence)
            rec_idx = len(records)
            records.append(
                {
                    "timestamp": round(frame_idx / fps, 2),
                    "emotion": emotion,
                    "confidence": round(confidence, 3),
                    "score": map_emotion_to_score(emotion),
                    "index_value": round(idx_val, 3),
                }
            )
            key_frames[rec_idx] = (frame.copy(), bbox)
            progress_bar.progress((step + 1) / n_samples)

        cap.release()
        tmp_path.unlink(missing_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — Webcam
# ══════════════════════════════════════════════════════════════════════════════

elif mode == "Webcam":
    st.title("Live Webcam Analysis")
