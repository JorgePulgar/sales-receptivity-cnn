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

st.title("Sales Receptivity CNN Demo")
