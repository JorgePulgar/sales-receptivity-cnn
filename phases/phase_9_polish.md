# Phase 9 — Final polish

## Objective

Ship-ready deliverable.

## Tasks

### Final README

Replace the bootstrap `README.md` with a full version including:

- **Project overview** — 2-3 paragraphs describing the business case and
  the system
- **Architecture diagram** — simple ASCII or a hand-drawn image showing
  data flow: video/webcam → face detection → CNN → emotion → receptivity
  index → output (API/UI)
- **Results summary** — table with the final model's test accuracy, F1
  macro, inference time. One sentence per metric.
- **Demo screenshots** — at least 3: receptivity timeline, session summary,
  webcam capture
- **Installation**
  - Prerequisites: Python 3.10, NVIDIA GPU with CUDA 11.2 + cuDNN 8.1
    (for GPU training) OR CPU-only mode
  - `python -m venv venv` and activation
  - `pip install -r requirements.txt`
- **Dataset setup**
  - Configure Kaggle API
  - `kaggle datasets download -d msambare/fer2013 -p data/raw/ --unzip`
- **Running the notebooks**
  - Order: 01 → 02 → 03 → 04
  - Approximate run times per notebook
- **Running the API**
  - `uvicorn src.api.main:app --reload`
  - Swagger UI URL
- **Running the demo**
  - `streamlit run demo/app.py`
- **Project structure** — tree of folders with one-line descriptions
- **Limitations and ethical considerations** — short version of the
  critical reflection from Notebook 4
- **License** — pick one (MIT is fine for a class project)

### End-to-end verification

In a separate clean folder:

1. Clone the project / copy it
2. Create fresh venv
3. `pip install -r requirements.txt`
4. Download FER2013 fresh
5. Run all 4 notebooks in order, top to bottom, no errors
6. Launch API, hit `/health`, run all tests
7. Launch demo, upload a sample video, capture from webcam

If anything fails: document the fix in the README (do not assume a future
developer will figure it out).

### Notebook cleanup

- Clear all `stderr` outputs (FutureWarnings, deprecation warnings) from
  cells where they are noise
- Keep meaningful outputs (training curves, summaries) but reduce
  verbose progress logs (`verbose=2` instead of `verbose=1` in `fit()`
  for cleaner notebooks)
- Verify there are no leftover scratch cells or debug prints
- Verify markdowns flow naturally — this is the manual review step where
  Claude Code's drafts must be rewritten by the user with personal voice
  and real decisions

### Repo hygiene

- `.gitignore` is complete (verify no `data/` or `models/*.keras`
  staged for commit)
- `requirements.txt` exhaustively covers what was actually used (test
  by reinstalling in a clean venv)
- No dead code in `src/`
- No commented-out experimental blocks left around

### Optional but recommended

- Export final model to TensorFlow Lite (`.tflite`) and document the
  conversion in the README. Mention inference latency improvement if
  measured.
- Record a 60-second screen capture of the demo and link it from the
  README.

## Validation checklist

- [ ] README renders correctly on GitHub or whichever platform is used
- [ ] All 4 notebooks run top to bottom in a fresh environment
- [ ] All tests pass
- [ ] API and demo launch without errors
- [ ] Manual markdown review completed
- [ ] No sensitive data or large binaries committed
- [ ] requirements.txt fully reproducible

## Output

Project ready to submit.
