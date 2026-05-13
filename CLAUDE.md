# CLAUDE.md

Context and rules for working on this project with Claude Code.

## Project context

Deep Learning class project. Emotional receptivity analysis system for sales
presentations using a CNN trained on FER2013. Deliverable: 4 pedagogical Jupyter
notebooks + reusable modules in `src/` + FastAPI service + Streamlit demo.

This is NOT a production project. Prioritize clarity, didactic value and
reproducibility over extreme robustness.

## Stack and environment

- OS: Windows 10/11 native (NOT WSL)
- Python: 3.10
- GPU: local NVIDIA (TF 2.10 is the last version with native GPU support on Windows)
- TensorFlow: 2.10.1 (Keras 2.10, included in TF)
- Dataset: FER2013 downloaded via Kaggle API

DO NOT upgrade TensorFlow beyond 2.10.x. It will break GPU support on Windows native.

## File structure

Defined in `project_structure.md`. Respect it strictly.

Additional rules:
- `src/` must be importable from the notebooks without side effects
  (do not train or load data on import)
- Saved models go in `models/` in `.keras` format (fallback to `.h5` if `.keras`
  fails on TF 2.10)
- Raw data in `data/raw/` (gitignored). Processed data in `data/processed/`
- Notebooks numbered `01_`, `02_`, `03_`, `04_` according to phase

## Python code conventions

- snake_case for functions and variables
- PascalCase for classes
- Type hints on functions in `src/` (not required in notebooks)
- Google-style docstrings on public functions of `src/`
- Imports ordered: stdlib, third-party, local
- Do not use `from x import *`
- Constants in UPPERCASE at the top of the file or in `src/config.py`
- Paths with `pathlib.Path`, not strings

## Jupyter notebook conventions

Strict pedagogical format:

- ONE code cell per concept. No 80-line cells
- Every code cell is PRECEDED by a markdown cell explaining:
  - WHAT is going to happen
  - WHY it is done this way
  - The decision taken and the alternative discarded when applicable
- Markdown length: 2-5 sentences. Neither a single line nor a huge paragraph
- Suggested structure: "Here [action]. This is needed because [reason].
  Decision: [parameter] because [justification]"
- Didactic but not condescending tone. Reader has basic ML knowledge but no
  prior exposure to FER2013 nor to the sales domain
- Avoid empty phrases like "let's explore the data". Go straight to the what and why
- Connect with the business case when applicable (e.g. when talking about
  latency, mention the real-time demo)
- Headings (##, ###) to structure large sections of the notebook
- Every notebook starts with an introduction markdown cell (objective + what
  will be done) and ends with a "Summary and link to the next notebook" cell

Markdown cells will be manually reviewed by the user. They must be close to
final quality, not placeholders.

## Training conventions

- Set random seed at the start of every notebook that trains
  (numpy, tensorflow, random)
- Default callbacks: `EarlyStopping(patience=7, restore_best_weights=True)`,
  `ReduceLROnPlateau(patience=3, factor=0.5)`, `ModelCheckpoint`
- Save `history` as JSON after every training run in `models/histories/`
  so plots can be regenerated without retraining
- Class weights computed with `sklearn.utils.class_weight.compute_class_weight`
- Verify GPU at the start with `tf.config.list_physical_devices('GPU')`

## Sensitive data and file management

- Do not commit `data/raw/` nor `data/processed/` to git
- Do not commit heavy `models/*.keras` to git (use Git LFS only if strictly
  needed; better avoid it in this project)
- `.gitignore` must include: `data/`, `models/*.keras`, `models/*.h5`,
  `.ipynb_checkpoints/`, `__pycache__/`, `*.pyc`, `venv/`, `.env`, `.kaggle/`
- Do NOT ignore `models/histories/*.json`: those training histories must be
  tracked so plots can be regenerated in Notebook 4 without retraining

## Dependencies

- DO NOT install new dependencies without asking first
- Every new dependency goes into `requirements.txt` with a pinned version
- If a dependency conflicts with TF 2.10, stop and ask before changing anything

## Git workflow

- Every discrete task gets its own commit. Do not batch unrelated work into a
  single commit
- Every commit message must end with a `Co-Authored-By: Claude <noreply@anthropic.com>`
  trailer so Claude is shown as a contributor
- Use conventional-style prefixes (`docs:`, `feat:`, `fix:`, `refactor:`,
  `deps:`, `chore:`) in commit subjects
- Do not push to any remote unless explicitly asked

## Communication

- Before implementing a full phase, briefly confirm the plan
- If you hit a technical problem (e.g. a library does not work with TF 2.10),
  stop and ask instead of improvising a workaround
- At the end of every phase, report what was done and what is missing
