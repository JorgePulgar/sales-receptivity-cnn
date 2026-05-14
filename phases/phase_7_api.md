# Phase 7 — FastAPI service

## Objective

REST API exposing the trained model. Reuses `src/inference/*` so no
duplicated logic.

## Tasks

### `src/api/main.py`

- [x] FastAPI app with lifespan event that loads the selected model at startup
  (so the load happens once, not per request)
- [x] Instantiate `FaceDetector`, `EmotionClassifier` and a `ReceptivityIndex`
  per session

### `src/api/schemas.py`

- [x] `PredictionResponse`:
  - `emotion: str`
  - `confidence: float`
  - `probabilities: Dict[str, float]`  # emotion → probability
  - `receptivity_signal: str`
  - `receptivity_score: float`
  - `face_detected: bool`
  - `bbox: Optional[Tuple[int, int, int, int]]`
  - `inference_time_ms: float`
- [x] `FrameAnalysis`: same as `PredictionResponse` plus `frame_index: int`
- [x] `SessionSummary`:
  - `dominant_emotion: str`
  - `mean_receptivity: float`
  - `time_in_each_state: Dict[str, float]`  # emotion → seconds (or frame count)
  - `peak_frame: int`
  - `valley_frame: int`
- [x] `SessionResponse`:
  - `frames_analysis: List[FrameAnalysis]`
  - `receptivity_index_over_time: List[float]`
  - `session_summary: SessionSummary`
- [x] `HealthResponse`:
  - `status: str`
  - `model_loaded: bool`
  - `model_path: str`
  - `inference_time_ms: float`  # measured on a dummy 48×48 image at startup

### Endpoints

- [x] `POST /predict/image`
  - Input: `multipart/form-data` with an image file
  - Steps: read bytes → decode with cv2 → detect face → if no face, return
    `face_detected=false`; else extract ROI → predict → map → return
- [x] `POST /predict/session`
  - Input: `multipart/form-data` with a video file (mp4/avi)
  - Steps: open with `cv2.VideoCapture` → iterate frames (sample 1 frame
    per N for speed; configurable, default 1 fps equivalent) → for each
    frame, run the per-image pipeline → update receptivity index →
    accumulate
  - Returns `SessionResponse`
  - Use a streaming response only if needed for large files; otherwise
    process fully and return at once
- [x] `GET /health`
  - Returns `HealthResponse`

### Error handling

- [x] 400 Bad Request: file not provided or wrong content-type
- [x] 422 Unprocessable Entity: file is not a valid image / video
- [x] 500 Internal Server Error with a clear message if model inference fails

### Tests in `tests/test_api.py`

Use `TestClient` from `fastapi.testclient` (sync). Do NOT use
`httpx.AsyncClient` + `pytest-asyncio` here — `pytest-asyncio` is not in
`requirements.txt` and the sync client is sufficient for these endpoints:

- [ ] `test_health()` — GET /health returns 200 with `model_loaded=True`
- [ ] `test_predict_image_valid()` — POST a known test-set image, expect 200
  and a valid `PredictionResponse`
- [ ] `test_predict_image_no_face()` — POST a black image, expect
  `face_detected=False`
- [ ] `test_predict_image_invalid()` — POST a text file, expect 422

## Validation

- `uvicorn src.api.main:app --reload` starts cleanly
- Swagger UI loads at `http://localhost:8000/docs`
- All tests pass: `pytest tests/test_api.py -v`

## Notes

- For `POST /predict/session`, big videos can be slow. Document a maximum
  recommended duration (e.g. 2-3 min) in the docstring.
- CORS: enable for local development with `allow_origins=["*"]` so the
  Streamlit demo can hit the API if we choose to do it that way (optional,
  since the demo can also import `src.inference` directly).

## Output

Working API with tests. Next phase: Streamlit demo.
