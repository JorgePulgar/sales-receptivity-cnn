# Phase 9 — Final polish

## Objective

Ship-ready deliverable.

**Ordering note:** phase 10 (TF.js web demo) runs BEFORE this phase, because
phase 9 polishes the README, screenshots, structure, and end-to-end check —
all of which must already include the public web demo by the time they are
written. If phase 10 has not been completed, pause here and finish it first.

## Tasks

### Final README

- [x] **Live Demo badge at the top of the README** — large, prominent link to
  the GitHub Pages URL (`https://<user>.github.io/sales-receptivity-cnn/`).
  Phrase: "Try it now in your browser — no install required."
- [ ] Project overview — 2-3 paragraphs describing the business case and the system
- [ ] Architecture diagram — simple ASCII or hand-drawn image showing data flow:
  video/webcam → face detection → CNN → emotion → receptivity index → output (API/UI)
- [ ] Results summary — table with the final model's test accuracy, F1 macro,
  inference time. One sentence per metric.
- [ ] Demo screenshots — at least 4: receptivity timeline, session summary,
  webcam capture (Streamlit), AND the web demo running live in a browser
  (full canvas + stats panel visible)
- [ ] Installation section:
  - **Lead with: "The public web demo needs no install. Local setup below is
    only for training and the full Streamlit demo."**
  - Prerequisites: Python 3.10, NVIDIA GPU with CUDA 11.2 + cuDNN 8.1
    (for GPU training) OR CPU-only mode
  - `python -m venv venv` and activation
  - `pip install -r requirements.txt`
- [ ] Dataset setup section:
  - Configure Kaggle API
  - `kaggle datasets download -d msambare/fer2013 -p data/raw/ --unzip`
- [ ] Running the notebooks section (order: 01 → 02 → 03 → 04, approximate run times)
- [ ] Running the API section (`uvicorn src.api.main:app --reload`, Swagger UI URL)
- [ ] Running the demos section — covers BOTH:
  - **Web demo (public):** open the GitHub Pages URL, allow camera, done.
  - **Streamlit demo (local):** `streamlit run demo/app.py`, supports video
    upload and continuous webcam stream.
- [ ] Project structure — tree of folders with one-line descriptions
  (must include `docs/` and `scripts/`)
- [ ] Limitations and ethical considerations — must include:
  - Web demo runs TF.js on the browser's WebGL/CPU. Predictions are similar
    but not bit-identical to the Python pipeline.
  - Web demo uses BlazeFace; Streamlit demo uses OpenCV Haar Cascades.
    Bounding boxes (and therefore predictions) will differ slightly between
    the two demos for the same input.
- [ ] License (MIT)

### End-to-end verification

- [ ] Clone/copy project to a separate clean folder
- [ ] Create fresh venv and `pip install -r requirements.txt`
- [ ] Download FER2013 fresh
- [ ] Run all 4 notebooks in order, top to bottom, no errors
- [ ] Launch API, hit `/health`, run `pytest tests/test_api.py -v`
- [ ] Launch Streamlit demo, upload a sample video, run continuous webcam mode
- [ ] **Open the public GitHub Pages URL in a fresh browser profile**
  (incognito ideally — no cached state), grant camera permission, verify
  emotions update and receptivity index moves
- [ ] Document any failures and fixes in the README

### Notebook cleanup

- [x] Clear all `stderr` outputs (FutureWarnings, deprecation warnings) from noisy cells
- [x] Reduce verbose training logs (`verbose=2` instead of `verbose=1` in `fit()`)
- [ ] Remove leftover scratch cells and debug prints
- [ ] Verify markdowns flow naturally — manual review step where Claude Code's
  drafts are rewritten with personal voice and real decisions

### Repo hygiene

- [ ] Verify `.gitignore` is complete (no `data/` or `models/*.keras` staged)
- [ ] Add a comment in `.gitignore` next to the model rules clarifying that
  `docs/models/` is intentionally tracked — the TF.js weights must ship
  with the GitHub Pages site
- [ ] Verify `requirements.txt` exhaustively covers what was actually used
  (test by reinstalling in a clean venv) — MUST include `tensorflowjs` from
  phase 10
- [ ] Remove dead code from `src/` — but do NOT remove `scripts/export_tfjs.py`;
  it is intentional infrastructure, not dead code
- [ ] Remove commented-out experimental blocks

### Optional but recommended

- [ ] Export final model to TensorFlow Lite (`.tflite`) and document the
  conversion in the README. Mention inference latency improvement if measured.
  **Lower priority now that phase 10 ships a deployable TF.js model — keep
  only if there is time to spare.**
- [ ] Record a 60-second screen capture of the **Streamlit demo** (recorded
  video upload flow + live webcam) and link it from the README. The web
  demo speaks for itself via the live link, so the screen capture should
  focus on features the web demo does not show (full video analysis,
  timeline, key frames, CSV export).

## Validation checklist

- [ ] README renders correctly on GitHub or whichever platform is used
- [ ] Live Demo badge resolves to a working GitHub Pages URL
- [ ] All 4 notebooks run top to bottom in a fresh environment
- [ ] All tests pass
- [ ] API and Streamlit demo launch without errors
- [ ] Public web demo loads, camera works, predictions update
- [ ] Manual markdown review completed
- [ ] No sensitive data or large binaries committed
- [ ] `requirements.txt` fully reproducible (includes `tensorflowjs`)

## Output

Project ready to submit.
