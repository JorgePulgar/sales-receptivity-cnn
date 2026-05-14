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


def _run_inference(image: np.ndarray) -> PredictionResponse:
    """Run face detection + emotion classification on one frame.

    FaceDetector and EmotionClassifier are module-level singletons loaded at
    startup. ReceptivityIndex is instantiated per session (stateful per call).
    """
    t0 = time.perf_counter()
    bbox = _detector.detect_largest(image)
    if bbox is None:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return PredictionResponse(
            emotion="neutral",
            confidence=0.0,
            probabilities={e: 0.0 for e in config.EMOTION_LABELS},
            receptivity_signal="No face detected",
            receptivity_score=5.0,
            face_detected=False,
            bbox=None,
            inference_time_ms=elapsed_ms,
        )
    roi = _detector.extract_roi(
        image, bbox, target_size=_model_input_size, to_grayscale=True
    )
    label, confidence, probs = _classifier.predict(roi)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return PredictionResponse(
        emotion=label,
        confidence=confidence,
        probabilities=dict(zip(config.EMOTION_LABELS, probs.tolist())),
        receptivity_signal=map_emotion_to_signal(label),
        receptivity_score=map_emotion_to_score(label),
        face_detected=True,
        bbox=bbox,
        inference_time_ms=elapsed_ms,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return API status and a warm-up inference latency measured on a dummy input."""
    if _classifier is None:
        return HealthResponse(
            status="ok",
            model_loaded=False,
            model_path="",
            inference_time_ms=0.0,
        )
    dummy = np.zeros((48, 48), dtype=np.uint8)
    t0 = time.perf_counter()
    _classifier.predict(dummy)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_path=str(_model_path),
        inference_time_ms=elapsed_ms,
    )


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)) -> PredictionResponse:
    """Detect the largest face in the uploaded image and return emotion analysis.

    Returns face_detected=False when no face is found — not an error condition.
    """
    if _classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded at startup.")
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported content type '{content_type}'. "
                "Upload a valid image file (jpeg, png, …)."
            ),
        )
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=422, detail="Could not decode file as an image.")
    try:
        return _run_inference(image)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Inference failed: {exc}"
        ) from exc
