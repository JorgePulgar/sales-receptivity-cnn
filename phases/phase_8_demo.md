# Phase 8 — Streamlit demo

## Objective

Visual demo of the system. Two modes: recorded video analysis and webcam
capture via `st.camera_input`. NO `streamlit-webrtc` — keep it robust.

## Tasks

### `demo/app.py` — Architecture

- [x] Top-level comment documenting the decision to import `src.inference`
  directly (not through the FastAPI endpoint), to avoid running two processes
  and keep the demo self-contained for the class presentation.

### Sidebar configuration

- [x] Title and short description
- [x] Selectbox: "Mode" → "Recorded video" or "Webcam"
- [x] Slider: receptivity index window size (default 10, range 5-30)
- [x] Checkbox: weight by confidence (default True)
- [x] Information block: which model is loaded (read from `config`)

### Mode 1 — Recorded video

- [x] `st.file_uploader` accepting `.mp4`, `.avi`, `.mov`
- [x] On upload: save to temp file, open with `cv2.VideoCapture`, read total
  frames and FPS, sample at 1 frame per second, show progress bar
- [x] For each sampled frame: detect face → if face: predict + update receptivity
  index; if no face: skip update but log
- [x] Tab 1 — Receptivity timeline:
  - Plotly line chart of receptivity index over time
  - Shaded background regions per dominant emotion segment
  - Annotations on top 3 peaks and bottom 3 valleys
  - Markdown caption explaining how to read the chart
- [x] Tab 2 — Session summary:
  - `st.metric` cards for: dominant emotion, mean receptivity, frames analyzed,
    frames without face
  - Pie chart or bar chart of time spent per emotion
- [x] Tab 3 — Frame-by-frame table:
  - `st.dataframe` with columns: timestamp, emotion, confidence, score, index_value
  - CSV download button
- [x] Tab 4 — Key moments:
  - 3 highest-receptivity frames with thumbnails and predicted emotion
  - 3 lowest-receptivity frames likewise

### Mode 2 — Webcam

- [x] `st.camera_input("Take a photo")` capture
- [x] On capture: detect face → predict → update receptivity index stored in
  `st.session_state`
- [x] Display captured photo with bounding box overlaid (`cv2.rectangle` + `st.image`)
- [x] Predicted emotion + confidence
- [ ] Current receptivity index as `st.metric`
- [ ] Cumulative emotion histogram (`st.bar_chart`)
- [ ] Receptivity index history (`st.line_chart`)
- [ ] `st.button("Reset session")` to clear `st.session_state` history

### State management and performance

- [ ] `st.session_state` guard for `ReceptivityIndex` in webcam mode:
  ```python
  if 'receptivity_index' not in st.session_state:
      st.session_state.receptivity_index = ReceptivityIndex(...)
  ```
- [ ] Wipe state when the user switches modes (detect via callback)
- [ ] Cache classifier and detector with `@st.cache_resource`:
  ```python
  @st.cache_resource
  def get_classifier():
      return EmotionClassifier(...)
  ```

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
- Frame processing in Mode 1 should not exceed 30 seconds for a 2-min video

## Output

Working demo. Next phase: final polish.
