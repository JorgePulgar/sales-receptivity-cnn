# Phase 9 — Final polish

## Objective

Ship-ready deliverable.

## Tasks

### Final README

- [ ] Project overview — 2-3 paragraphs describing the business case and the system
- [ ] Architecture diagram — simple ASCII or hand-drawn image showing data flow:
  video/webcam → face detection → CNN → emotion → receptivity index → output (API/UI)
- [ ] Results summary — table with the final model's test accuracy, F1 macro,
  inference time. One sentence per metric.
- [ ] Demo screenshots — at least 3: receptivity timeline, session summary,
  webcam capture
- [ ] Installation section:
  - Prerequisites: Python 3.10, NVIDIA GPU with CUDA 11.2 + cuDNN 8.1
    (for GPU training) OR CPU-only mode
  - `python -m venv venv` and activation
  - `pip install -r requirements.txt`
- [ ] Dataset setup section:
  - Configure Kaggle API
  - `kaggle datasets download -d msambare/fer2013 -p data/raw/ --unzip`
- [ ] Running the notebooks section (order: 01 → 02 → 03 → 04, approximate run times)
- [ ] Running the API section (`uvicorn src.api.main:app --reload`, Swagger UI URL)
- [ ] Running the demo section (`streamlit run demo/app.py`)
- [ ] Project structure — tree of folders with one-line descriptions
- [ ] Limitations and ethical considerations
- [ ] License (MIT)

### End-to-end verification

- [ ] Clone/copy project to a separate clean folder
- [ ] Create fresh venv and `pip install -r requirements.txt`
- [ ] Download FER2013 fresh
- [ ] Run all 4 notebooks in order, top to bottom, no errors
- [ ] Launch API, hit `/health`, run `pytest tests/test_api.py -v`
- [ ] Launch demo, upload a sample video, capture from webcam
- [ ] Document any failures and fixes in the README

### Notebook cleanup

- [ ] Clear all `stderr` outputs (FutureWarnings, deprecation warnings) from noisy cells
- [ ] Reduce verbose training logs (`verbose=2` instead of `verbose=1` in `fit()`)
- [ ] Remove leftover scratch cells and debug prints
- [ ] Verify markdowns flow naturally — manual review step where Claude Code's
  drafts are rewritten with personal voice and real decisions

### Repo hygiene

- [ ] Verify `.gitignore` is complete (no `data/` or `models/*.keras` staged)
- [ ] Verify `requirements.txt` exhaustively covers what was actually used
  (test by reinstalling in a clean venv)
- [ ] Remove dead code from `src/`
- [ ] Remove commented-out experimental blocks

### Optional but recommended

- [ ] Export final model to TensorFlow Lite (`.tflite`) and document the
  conversion in the README. Mention inference latency improvement if measured.
- [ ] Record a 60-second screen capture of the demo and link it from the README.

## Validation checklist

- [ ] README renders correctly on GitHub or whichever platform is used
- [ ] All 4 notebooks run top to bottom in a fresh environment
- [ ] All tests pass
- [ ] API and demo launch without errors
- [ ] Manual markdown review completed
- [ ] No sensitive data or large binaries committed
- [ ] `requirements.txt` fully reproducible

## Output

Project ready to submit.
