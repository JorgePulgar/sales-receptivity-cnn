"""Tests for src/inference modules.

FaceDetector: interface tested on a blank image (no crash, correct return
types) plus an optional real-face test that is skipped when FER2013 raw
data is absent.

ReceptivityIndex: feed known emotion sequences and assert the index moves
in the expected direction.

EmotionClassifier: skipped when no model file is present on disk.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.inference.face_detector import FaceDetector
from src.inference.receptivity_mapper import (
    ReceptivityIndex,
    map_emotion_to_score,
    map_emotion_to_signal,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_fer_image() -> Path | None:
    """Return the path to any single image from the FER2013 raw train split."""
    train_dir = config.RAW_DIR / "train"
    if not train_dir.exists():
        return None
    for emotion_dir in sorted(train_dir.iterdir()):
        if not emotion_dir.is_dir():
            continue
        for img_path in emotion_dir.iterdir():
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                return img_path
    return None


# ---------------------------------------------------------------------------
# FaceDetector — interface tests (no real face needed)
# ---------------------------------------------------------------------------

class TestFaceDetectorInterface:
    def setup_method(self):
        self.detector = FaceDetector()

    def test_instantiation(self):
        assert self.detector is not None

    def test_detect_returns_list_on_blank_image(self):
        blank = np.zeros((100, 100), dtype=np.uint8)
        result = self.detector.detect(blank)
        assert isinstance(result, list)

    def test_detect_largest_returns_none_on_blank_image(self):
        blank = np.zeros((100, 100), dtype=np.uint8)
        result = self.detector.detect_largest(blank)
        assert result is None

    def test_extract_roi_shape(self):
        image = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        bbox = (10, 10, 50, 50)
        roi = self.detector.extract_roi(image, bbox, target_size=(48, 48), to_grayscale=True)
        assert roi.shape == (48, 48)
        assert roi.dtype == np.uint8

    def test_extract_roi_rgb_shape(self):
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        bbox = (10, 10, 50, 50)
        roi = self.detector.extract_roi(image, bbox, target_size=(48, 48), to_grayscale=False)
        assert roi.shape == (48, 48, 3)


@pytest.mark.skipif(
    _first_fer_image() is None,
    reason="FER2013 raw data not present — skipping real face-detection test",
)
class TestFaceDetectorWithRealImage:
    def setup_method(self):
        self.detector = FaceDetector()
        import cv2
        img_path = _first_fer_image()
        self.image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    def test_detect_returns_list(self):
        result = self.detector.detect(self.image)
        assert isinstance(result, list)

    def test_detect_largest_returns_tuple_or_none(self):
        result = self.detector.detect_largest(self.image)
        assert result is None or (isinstance(result, tuple) and len(result) == 4)


# ---------------------------------------------------------------------------
# ReceptivityIndex
# ---------------------------------------------------------------------------

class TestReceptivityIndex:
    def test_initial_index_is_neutral(self):
        ri = ReceptivityIndex()
        assert ri.get_current_index() == pytest.approx(5.0)

    def test_happy_observations_raise_index(self):
        ri = ReceptivityIndex(window_size=5, weight_by_confidence=False)
        for _ in range(5):
            ri.update("happy", confidence=0.9)
        assert ri.get_current_index() > 5.0

    def test_angry_observations_lower_index(self):
        ri = ReceptivityIndex(window_size=5, weight_by_confidence=False)
        for _ in range(5):
            ri.update("angry", confidence=0.9)
        assert ri.get_current_index() < 5.0

    def test_mixed_sequence_moves_in_expected_direction(self):
        ri = ReceptivityIndex(window_size=10, weight_by_confidence=False)
        # Fill with neutral first
        for _ in range(5):
            ri.update("neutral", 0.8)
        neutral_index = ri.get_current_index()
        # Then push happy
        for _ in range(5):
            ri.update("happy", 0.9)
        assert ri.get_current_index() > neutral_index

    def test_history_length_matches_update_count(self):
        ri = ReceptivityIndex()
        emotions = ["happy", "sad", "neutral", "angry", "surprise"]
        for e in emotions:
            ri.update(e, 0.8)
        assert len(ri.get_history()) == len(emotions)

    def test_reset_clears_state(self):
        ri = ReceptivityIndex()
        ri.update("happy", 0.9)
        ri.reset()
        assert ri.get_current_index() == pytest.approx(5.0)
        assert ri.get_history() == []

    def test_window_size_respected(self):
        ri = ReceptivityIndex(window_size=3, weight_by_confidence=False)
        ri.update("angry", 1.0)
        ri.update("angry", 1.0)
        ri.update("angry", 1.0)
        # Now push three happy — angry should be evicted
        ri.update("happy", 1.0)
        ri.update("happy", 1.0)
        ri.update("happy", 1.0)
        assert ri.get_current_index() == pytest.approx(map_emotion_to_score("happy"))

    def test_weighted_confidence_gives_high_confidence_more_weight(self):
        ri = ReceptivityIndex(window_size=2, weight_by_confidence=True)
        # Low-confidence angry vs high-confidence happy
        ri.update("angry", confidence=0.1)
        ri.update("happy", confidence=0.9)
        # Weighted average should be closer to happy score
        assert ri.get_current_index() > 5.0


# ---------------------------------------------------------------------------
# Map functions
# ---------------------------------------------------------------------------

class TestMapFunctions:
    def test_map_emotion_to_score_known_emotions(self):
        for emotion in config.EMOTION_LABELS:
            score = map_emotion_to_score(emotion)
            assert 0.0 <= score <= 10.0

    def test_map_emotion_to_signal_returns_string(self):
        for emotion in config.EMOTION_LABELS:
            signal = map_emotion_to_signal(emotion)
            assert isinstance(signal, str) and len(signal) > 0

    def test_happy_has_higher_score_than_angry(self):
        assert map_emotion_to_score("happy") > map_emotion_to_score("angry")


# ---------------------------------------------------------------------------
# EmotionClassifier — skipped when no model file is present
# ---------------------------------------------------------------------------

_MODEL_PATH = config.MODELS_DIR / "cnn_custom.keras"
_MODEL_PATH_H5 = config.MODELS_DIR / "cnn_custom.h5"
_model_available = _MODEL_PATH.exists() or _MODEL_PATH_H5.exists()


@pytest.mark.skipif(
    not _model_available,
    reason="No trained model file found — skipping EmotionClassifier test",
)
class TestEmotionClassifier:
    def setup_method(self):
        from src.inference.emotion_classifier import EmotionClassifier

        path = _MODEL_PATH if _MODEL_PATH.exists() else _MODEL_PATH_H5
        self.clf = EmotionClassifier(
            model_path=path,
            input_size=config.IMG_SIZE_CUSTOM,
            use_rgb=False,
        )

    def test_predict_returns_label_confidence_probs(self):
        face = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
        label, conf, probs = self.clf.predict(face)
        assert label in config.EMOTION_LABELS
        assert 0.0 <= conf <= 1.0
        assert probs.shape == (len(config.EMOTION_LABELS),)
        assert abs(probs.sum() - 1.0) < 1e-4

    def test_predict_batch_length_matches_input(self):
        faces = np.random.randint(0, 255, (4, 48, 48), dtype=np.uint8)
        results = self.clf.predict_batch(faces)
        assert len(results) == 4
