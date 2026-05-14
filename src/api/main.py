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


@app.post("/predict/session", response_model=SessionResponse)
async def predict_session(file: UploadFile = File(...)) -> SessionResponse:
    """Analyse a video file and return per-frame emotion analysis with a session summary.

    Frames are sampled at 1 fps (SAMPLE_FPS). Recommended maximum duration: 2-3 minutes;
    longer videos increase processing time proportionally.
    """
    SAMPLE_FPS = 1

    if _classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded at startup.")

    content_type = file.content_type or ""
    if not (
        content_type.startswith("video/") or content_type == "application/octet-stream"
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported content type '{content_type}'. "
                "Upload a valid video file (mp4, avi, …)."
            ),
        )

    data = await file.read()
    tmp_path = Path(tempfile.mktemp(suffix=".mp4"))
    tmp_path.write_bytes(data)

    cap = cv2.VideoCapture(str(tmp_path))
    if not cap.isOpened():
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="Could not open file as a video.")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(fps / SAMPLE_FPS)))

        ri = ReceptivityIndex()
        frames_analysis: List[FrameAnalysis] = []
        receptivity_values: List[float] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                try:
                    pred = _run_inference(frame)
                except Exception as exc:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Inference failed on frame {frame_idx}: {exc}",
                    ) from exc
                if pred.face_detected:
                    ri_val = ri.update(pred.emotion, pred.confidence)
                else:
                    ri_val = ri.get_current_index()
                receptivity_values.append(ri_val)
                frames_analysis.append(
                    FrameAnalysis(**pred.model_dump(), frame_index=frame_idx)
                )
            frame_idx += 1
    finally:
        cap.release()
        tmp_path.unlink(missing_ok=True)

    if not frames_analysis:
        raise HTTPException(
            status_code=422, detail="No frames could be extracted from the video."
        )

    valid_frames = [f for f in frames_analysis if f.face_detected]
    if valid_frames:
        emotion_counts = Counter(f.emotion for f in valid_frames)
        dominant_emotion = emotion_counts.most_common(1)[0][0]
        mean_receptivity = float(
            np.mean([map_emotion_to_score(f.emotion) for f in valid_frames])
        )
        time_in_each_state = {e: float(c) for e, c in emotion_counts.items()}
        scores_indexed = [
            (map_emotion_to_score(f.emotion), f.frame_index) for f in valid_frames
        ]
        peak_frame = max(scores_indexed, key=lambda x: x[0])[1]
        valley_frame = min(scores_indexed, key=lambda x: x[0])[1]
    else:
        dominant_emotion = "neutral"
        mean_receptivity = 5.0
        time_in_each_state = {}
        peak_frame = frames_analysis[0].frame_index
        valley_frame = frames_analysis[0].frame_index

    return SessionResponse(
        frames_analysis=frames_analysis,
        receptivity_index_over_time=receptivity_values,
        session_summary=SessionSummary(
            dominant_emotion=dominant_emotion,
            mean_receptivity=mean_receptivity,
            time_in_each_state=time_in_each_state,
            peak_frame=peak_frame,
            valley_frame=valley_frame,
        ),
    )
