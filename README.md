# Sales Receptivity CNN

CNN-based emotional receptivity analysis system for sales presentations.
Detects facial expressions from webcam or recorded video and produces a
dynamic receptivity index along with a post-session coaching report.

## Project documents

- `CLAUDE.md` — context and rules for Claude Code
- `requirements.txt` — pinned dependencies (TF 2.10 on Windows native + GPU)
- `phases/` — phase-by-phase implementation plan
  - `phase_0_setup.md`
  - `phase_1_base_modules.md`
  - `phase_2_notebook_eda.md`
  - `phase_3_notebook_preprocessing.md`
  - `phase_4_inference_modules.md`
  - `phase_5_notebook_training.md`
  - `phase_6_notebook_evaluation.md`
  - `phase_7_api.md`
  - `phase_8_demo.md`
  - `phase_9_polish.md`

## How to use these documents

1. Drop them in the root of an empty project folder
2. Start Claude Code in that folder
3. Feed phases one by one (do NOT paste all of them at once)
4. At the end of each phase, review the output before moving to the next

## Stack summary

- Python 3.10
- TensorFlow 2.10.1 (GPU on Windows native)
- FastAPI + Streamlit
- FER2013 dataset via Kaggle
