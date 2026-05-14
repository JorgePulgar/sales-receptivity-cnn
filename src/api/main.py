"""FastAPI service exposing the emotion-receptivity inference pipeline.

Start with:
    uvicorn src.api.main:app --reload

Swagger UI: http://localhost:8000/docs
"""

import sys
import tempfile
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.api.schemas import (
    FrameAnalysis,
    HealthResponse,
    PredictionResponse,
    SessionResponse,
    SessionSummary,
)
from src.inference.emotion_classifier import EmotionClassifier
from src.inference.face_detector import FaceDetector
from src.inference.receptivity_mapper import (
    ReceptivityIndex,
    map_emotion_to_score,
    map_emotion_to_signal,
)

_detector: Optional[FaceDetector] = None
_classifier: Optional[EmotionClassifier] = None
_model_path: Optional[Path] = None
_model_input_size: Optional[tuple] = None


def _find_model() -> Optional[Path]:
    """Return the first available trained model from models/."""
    for candidate in [
        config.MODELS_DIR / "custom_cnn.keras",
        config.MODELS_DIR / "custom_cnn.h5",
        config.MODELS_DIR / "mobilenet_finetune.keras",
        config.MODELS_DIR / "mobilenet_finetune.h5",
    ]:
        if candidate.exists():
            return candidate
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the face detector and emotion classifier once at startup."""
    global _detector, _classifier, _model_path, _model_input_size
    _detector = FaceDetector()
    path = _find_model()
    if path is not None:
        _model_path = path
        use_rgb = "mobilenet" in path.stem
        _model_input_size = config.IMG_SIZE_MOBILENET if use_rgb else config.IMG_SIZE_CUSTOM
        _classifier = EmotionClassifier(
            model_path=path,
            input_size=_model_input_size,
            use_rgb=use_rgb,
        )
    yield


app = FastAPI(
    title="Sales Receptivity API",
    description="Emotion analysis from images and video for sales presentations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
