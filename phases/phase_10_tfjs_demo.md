# Phase 10 — TF.js web demo (GitHub Pages)

## Objective

Deploy a public, real-time webcam demo that any portfolio visitor can open
via a link in the README. The browser accesses the camera natively and runs
inference client-side using TensorFlow.js — no server required, no install.

The Streamlit demo (`demo/app.py`) stays untouched for local use and the class
presentation. This phase is purely additive.

## Tasks

### Model export to TF.js

- [x] Check `tensorflowjs` compatibility with TF 2.10 and pin a version in
  `requirements.txt` (`tensorflowjs==4.x.x` should work; do not upgrade TF)
- [x] Write `scripts/export_tfjs.py`:
  - Load `models/mobilenet_ft.keras` (or `.h5` fallback) and convert to
    TF.js **layers model** format → `docs/models/mobilenet_ft/`
  - Load `models/cnn_custom.keras` (or `.h5` fallback) and convert →
    `docs/models/cnn_custom/`
  - Print input shape and output shape for each model as a sanity check
  - Decision: layers model (direct from `.keras`) over graph model. Graph
    needs a SavedModel intermediate; layers loads via `tf.loadLayersModel`
    in JS and is fast enough for 47 MB of weights. Revisit only if browser
    inference is the bottleneck.
- [ ] Commit the exported `docs/models/` files to git — this is intentional
  and the only exception to the "do not commit models" rule. Total size is
  ~47 MB (cnn_custom 19.6 MB + mobilenet_ft 27.8 MB); GitHub Pages serves
  them directly; Git LFS is not needed.
- [x] **Smoke test before building UI:** write a minimal `docs/smoke.html`
  that loads one model and predicts on a hardcoded test image, logging the
  softmax vector to the console. Verify it matches the Python prediction on
  the same image within a small tolerance. Decouples model-export bugs from
  UI bugs. Delete or hide this page once the real demo works.

### Face detection in the browser

- [ ] Use `@tensorflow-models/face-detection` (BlazeFace backend) loaded from
  CDN — same TF.js runtime as the emotion model, no dependency conflict.
  Decision: BlazeFace over face-api.js (lighter, no secondary TF.js instance)
  and over MediaPipe (no WASM setup required).
- [ ] Wrap detection in a `detectFace(videoEl)` function that returns the
  bounding box of the largest face (or `null`).
- [ ] **Document the detector mismatch with the Streamlit demo:** local uses
  OpenCV Haar Cascades, web uses BlazeFace. Bounding boxes differ → cropped
  ROIs differ → predictions will not be bit-identical between the two demos.
  This is acceptable; note it in the README's limitations section so it is
  not later misdiagnosed as a model bug.

### Real-time inference loop (`docs/app.js`)

- [ ] Access camera via `navigator.mediaDevices.getUserMedia({ video: true })`
- [ ] Draw the video stream to a `<canvas>` at ~8 fps using
  `requestAnimationFrame` throttled by elapsed time
- [ ] Per frame pipeline:
  1. `detectFace` → get bounding box
  2. Extract ROI from canvas as `ImageData`
  3. Preprocess:
     - MobileNetV2: resize to 96×96, normalise [0, 1], keep 3 channels (RGB)
     - Custom CNN: resize to 48×48, normalise [0, 1], convert to grayscale
       using ITU-R BT.601 (`0.299*R + 0.587*G + 0.114*B` — matches OpenCV's
       `cvtColor(RGB→GRAY)`), add channel dimension
  4. `model.predict(tensor)` → softmax vector
  5. Argmax → emotion label + confidence
  6. Update receptivity index (rolling window, reimplemented in JS)
  7. Redraw canvas overlay (bounding box + label)
  8. Update UI metrics
- [ ] Wrap the per-frame pipeline in `tf.tidy()` so intermediate tensors are
  auto-disposed. Manual `dispose()` is error-prone — `tf.tidy()` covers the
  whole scope. Only the final softmax vector needs `.array()`-ing out before
  the tidy block ends.

### Receptivity index in JS

- [ ] Reimplement `ReceptivityIndex` as a plain JS class:
  - Same emotion → score mapping as `src/inference/receptivity_mapper.py`
  - Same rolling weighted-average logic (window size configurable)
  - No external library needed

### UI (`docs/index.html` + `docs/style.css`)

- [ ] Video canvas (left) + stats panel (right) layout
- [ ] Model selector dropdown: "MobileNetV2 fine-tuned" / "Custom CNN" —
  switching unloads the current model and loads the new one without page reload
- [ ] Window-size slider (5–30, default 10), same as the Streamlit sidebar
- [ ] Emotion badge: label + confidence percentage, colour-coded per emotion
- [ ] Receptivity index metric: large number + 0–10 scale bar
- [ ] Rolling receptivity chart: last 30 s of values (plain `<canvas>` with
  manual draw — no Chart.js dependency to keep load time minimal)
- [ ] Emotion frequency bar chart (same approach)
- [ ] "No face detected" indicator
- [ ] "Camera access denied" error state with instructions

### GitHub Pages setup

- [ ] Add `docs/.nojekyll` (empty file — prevents Jekyll from processing the
  folder and mangling filenames with underscores)
- [ ] Enable GitHub Pages in repo settings: source = `docs/` branch `main`
- [ ] Verify the page loads at
  `https://<username>.github.io/sales-receptivity-cnn/`

### README update (done last)

- [ ] Add a "Live Demo" badge/button at the top of the README linking to the
  GitHub Pages URL
- [ ] Add one screenshot or short GIF of the web demo in action

## Notes

- The JS preprocessing must exactly match the Python `_preprocess` method in
  `EmotionClassifier` — pixel normalisation, channel order, and resize
  interpolation all matter for consistent predictions.
- `docs/models/` is intentionally tracked. Current `.gitignore` ignores
  `models/*.keras` and `models/*.h5`, which does not match `docs/models/`,
  so no rule change is required — but add a comment in `.gitignore` next to
  the existing model rules so this is visible.
- **GitHub Pages subpath:** the site lives at
  `https://<user>.github.io/sales-receptivity-cnn/`. All asset references
  in HTML/JS MUST be relative (`models/mobilenet_ft/model.json`,
  `app.js`, `style.css`) — absolute paths like `/models/...` resolve to
  `<user>.github.io/models/...` and 404.
- **HTTPS for camera:** `getUserMedia` only works in a secure context.
  GitHub Pages serves HTTPS by default, so this is automatic — only worth
  noting if anyone tries to test the page via `file://` and wonders why the
  camera prompt never appears.
- Do not add `Chart.js` or other heavy JS dependencies — the demo page must
  load in under 3 seconds on a typical connection.

## Validation

- [ ] `python scripts/export_tfjs.py` completes without error and outputs
  `model.json` + weight shards for each model under `docs/models/`
- [ ] Page loads in Chrome and Firefox without console errors
- [ ] Camera permission prompt appears; stream renders in canvas
- [ ] Face bounding box overlays correctly on the video
- [ ] Emotion label and confidence update at ~8 fps
- [ ] Receptivity index matches the Streamlit output for the same video content
  (within rounding; compare manually with a static image)
- [ ] Model switch works without page reload
- [ ] GitHub Pages URL is publicly accessible without login

## Output

Public demo URL committed to README. Any visitor can open the link, allow
camera access, and see real-time emotion and receptivity analysis — no install,
no Python, no GPU required.
