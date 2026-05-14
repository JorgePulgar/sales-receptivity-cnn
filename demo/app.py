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

st.title("Sales Receptivity CNN Demo")
