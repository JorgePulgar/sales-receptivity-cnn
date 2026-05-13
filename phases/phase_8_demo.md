# Phase 8 — Streamlit demo

## Objective

Visual demo of the system. Two modes: recorded video analysis and webcam
capture via `st.camera_input`. NO `streamlit-webrtc` — keep it robust.

## Tasks

### `demo/app.py`

**Architecture decision:** the demo imports `src.inference` directly,
NOT through the FastAPI endpoint. This avoids running two processes and
keeps the demo self-contained for the class presentation. Document this
in a top-level comment.

### Sidebar configuration

- Title and short description
- Selectbox: "Mode" → "Recorded video" or "Webcam"
- Slider: receptivity index window size (default 10, range 5-30)
- Checkbox: weight by confidence (default True)
- Information block: which model is loaded (read from `config`)

### Mode 1 — Recorded video (implement FIRST)

- `st.file_uploader` accepting `.mp4`, `.avi`, `.mov`
- On upload:
  - Save to a temp file
  - Open with `cv2.VideoCapture`
  - Read total frames and FPS
  - Sample frames at 1 frame per second of video (configurable)
  - Show a progress bar while iterating
  - For each sampled frame:
    - Detect face
    - If face: predict + update receptivity index
    - If no face: skip update but log
  - Collect: list of `(timestamp, emotion, confidence, score, index_value)`

After processing, show in tabs or expandable sections:

**Tab 1 — Receptivity timeline:**
- Plotly line chart of receptivity index over time
- Shaded background regions per dominant emotion segment
- Annotations on top 3 peaks and bottom 3 valleys
- Markdown caption explaining how to read the chart

**Tab 2 — Session summary:**
- `st.metric` cards for: dominant emotion, mean receptivity, frames
  analyzed, frames without face
- Pie chart or bar chart of time spent per emotion

**Tab 3 — Frame-by-frame table:**
- `st.dataframe` with all sampled frames
- Columns: timestamp, emotion, confidence, score, index_value
- Allow CSV download

**Tab 4 — Key moments:**
- Show the 3 highest-receptivity frames with thumbnails and predicted emotion
- Show the 3 lowest-receptivity frames likewise

### Mode 2 — Webcam (`st.camera_input`)

- `st.camera_input("Take a photo")` returns a single image
- On capture:
  - Detect face
  - Predict
  - Update receptivity index (state stored in `st.session_state`)
- Display:
  - Captured photo with bounding box overlaid (use `cv2.rectangle` + show
    via `st.image`)
  - Predicted emotion + confidence
  - Current receptivity index as `st.metric`
  - Cumulative emotion histogram (`st.bar_chart`)
  - Receptivity index history (`st.line_chart`)
- `st.button("Reset session")` to clear `st.session_state` history

### State management

- Use `st.session_state` for the `ReceptivityIndex` instance in webcam mode
- Initialize on first run with a guard:
  ```python
  if 'receptivity_index' not in st.session_state:
      st.session_state.receptivity_index = ReceptivityIndex(...)
  ```
- Wipe state when the user switches modes (detect via callback)

### Performance considerations

- Cache the classifier and detector with `@st.cache_resource`:
  ```python
  @st.cache_resource
  def get_classifier():
      return EmotionClassifier(...)
  ```
- Frame processing in Mode 1 should not exceed 30 seconds for a 2-min video

## Validation

- `streamlit run demo/app.py` opens the UI without errors
- Upload a 30-second test video and verify all tabs populate
- Use webcam capture and verify the index updates

## Notes

- Record at least one 2-minute demo video of yourself making exaggerated
  expressions (happy, surprise, neutral, sad, angry) and keep it under
  `demo/sample_videos/` (gitignored) for live testing during the class
  presentation
- If the file is too large to demo live, prepare a screenshot or short
  GIF as a fallback

## Output

Working demo. Next phase: final polish.
