# Phase 4 — Inference modules in `src/inference`

## Objective

Reusable inference logic that will be consumed by the API and the Streamlit
demo. Must work with a dummy model for testing before training is finished.

## Tasks

### `src/inference/face_detector.py`

- [x] Class `FaceDetector`:
  - `__init__(self, scale_factor: float = 1.1, min_neighbors: int = 5,
     min_size: Tuple[int, int] = (30, 30))`
  - Loads `cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')`
- [x] `detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]`
  returning bounding boxes `(x, y, w, h)`
- [x] `detect_largest(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]`
  returning only the largest detected face (most relevant during a
  sales call: the person on screen)
- [x] `extract_roi(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
   target_size: Tuple[int, int], to_grayscale: bool = True) -> np.ndarray`
  crops and resizes the face ROI

### `src/inference/emotion_classifier.py`

- [ ] Class `EmotionClassifier`:
  - `__init__(self, model_path: Path, input_size: Tuple[int, int],
     use_rgb: bool, labels: List[str] = config.EMOTION_LABELS)`
  - Loads the Keras model from disk at construction time
- [ ] `predict(self, face_roi: np.ndarray) -> Tuple[str, float, np.ndarray]`
  returning `(emotion_label, confidence, probabilities_vector)`
  - Handles preprocessing internally: normalization, channel duplication
    if `use_rgb=True`, batch axis
- [ ] `predict_batch(self, face_rois: np.ndarray) -> List[Tuple[str, float, np.ndarray]]`
  for video processing

### `src/inference/receptivity_mapper.py`

- [ ] `map_emotion_to_signal(emotion: str) -> str` returns text like
  "high interest", "passive attention", "active resistance"
- [ ] `map_emotion_to_score(emotion: str) -> float` simple lookup
- [ ] Class `ReceptivityIndex`:
  - `__init__(self, window_size: int = 10, weight_by_confidence: bool = True)`
  - `update(self, emotion: str, confidence: float) -> float`
    appends to internal buffer, returns current weighted moving average
  - `get_current_index(self) -> float`
  - `reset(self) -> None`
  - `get_history(self) -> List[float]` for plotting
  - Behavior on no face detected: the caller should NOT call `update`; the
    index keeps its last value. Document this contract in the docstring.

### Tests

- [ ] `tests/test_inference.py`:
  - Test for `FaceDetector` using a known sample image with a face
  - Test for `ReceptivityIndex`: feed a sequence of emotions, assert the
    index moves in the expected direction
  - Test for `EmotionClassifier` skipped (`@pytest.mark.skipif` if no model
    file exists), or use a dummy random model

## Validation

- `pytest tests/test_inference.py` passes
- `FaceDetector` can detect faces in a sample image
- `ReceptivityIndex` increases when fed Happy and decreases when fed Angry

## Notes

- Do not introduce MediaPipe unless OpenCV Haar Cascades fail. Haar is more
  than enough for the demo and avoids extra dependencies that occasionally
  give issues on Windows.
- These modules should be importable without GPU and without a trained model
  (constructor-time errors are OK if the model file is required at init).

## Output

Inference pipeline ready. Next phase: train the models (Notebook 3).
