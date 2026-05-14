"""Tests for the FastAPI endpoints.

Uses TestClient (sync) from fastapi.testclient. All tests are skipped when no
trained model file is present on disk.
"""

import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config

_MODEL_CANDIDATES = [
    config.MODELS_DIR / "custom_cnn.keras",
    config.MODELS_DIR / "custom_cnn.h5",
    config.MODELS_DIR / "mobilenet_finetune.keras",
    config.MODELS_DIR / "mobilenet_finetune.h5",
]
_model_available = any(p.exists() for p in _MODEL_CANDIDATES)

pytestmark = pytest.mark.skipif(
    not _model_available,
    reason="No trained model found — skipping API tests",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.api.main import app
    with TestClient(app) as c:
        yield c


def _encode_jpeg(array: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", array)
    return buf.tobytes()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "model_path" in data
    assert "inference_time_ms" in data


def test_predict_image_valid(client):
    # A solid-colour image is valid even if no face is detected; checks response shape.
    image_bytes = _encode_jpeg(np.zeros((100, 100, 3), dtype=np.uint8))
    response = client.post(
        "/predict/image",
        files={"file": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "emotion" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert "face_detected" in data
    assert "inference_time_ms" in data
