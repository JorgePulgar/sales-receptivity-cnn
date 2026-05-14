# Phase 6 — Notebook 4: Evaluation (`notebooks/04_evaluation.ipynb`)

## Objective

Full quantitative and qualitative analysis of both models. Select the final
model with documented justification. Produce the receptivity index simulation.

## Tasks

- [x] Section 0 — Introduction
  - Markdown: objective. List what will be produced (curves, confusion matrices,
    classification reports, Grad-CAM, receptivity simulation, final selection).

- [x] Section 1 — Setup
  - Imports
  - Load test data from `data/processed/fer2013_gray.npz` and
    `data/processed/fer2013_rgb64.npz`
  - Load both trained models from `models/`
  - Load both histories from `models/histories/`
  - Markdown: we never retrain here; pure evaluation

- [x] Section 2 — Training curves
  - For each model: plot loss and accuracy curves (train vs val) side by side
  - Mark the epoch of `EarlyStopping` restoration
  - Markdown per plot: diagnose overfitting (large train-val gap), underfitting
    (both curves low) or healthy convergence

- [x] Section 3 — Test set predictions
  - Predict probabilities on the test set for both models
  - Compute predicted class via argmax
  - Save predictions in memory for the rest of the notebook
  - Markdown: this is the single inference pass we will reuse downstream

- [x] Section 4 — Confusion matrices
  - Normalized confusion matrix for each model (`normalize='true'`)
  - Heatmap with class names on axes
  - Markdown: highlight expected confusions from the EDA (Fear↔Surprise,
    Sad↔Neutral, Angry↔Disgust). Confirm or contradict the hypothesis

- [x] Section 5 — Classification report
  - `sklearn.metrics.classification_report(..., output_dict=True)` for each model
  - Render as a DataFrame with precision, recall, F1 and support per class
  - Markdown: identify worst-performing classes (likely Disgust, Fear) and
    comment on whether class weights helped

- [ ] Section 6 — Comparative summary table
  - Build a DataFrame with one row per model:
    - Train accuracy (from history, last epoch before early stop)
    - Validation accuracy (same)
    - Test accuracy
    - F1 macro
    - Parameter count (`model.count_params()`)
    - Inference time per frame (measure on CPU: average 100 single-frame
      predictions with `time.perf_counter()`)
  - Render the table
  - Markdown: discuss tradeoffs

- [ ] Section 7 — Error analysis
  - For the better model, collect misclassified test examples
  - For each pair (true class, predicted class) that is most frequent, show
    a 4-image grid of examples
  - Markdown: are there systematic patterns? Lighting, occlusion, ambiguous
    expressions?

- [ ] Section 8 — Grad-CAM visualization
  - Use `tf-keras-vis` (`GradcamPlusPlus`) on the better model
  - For each of the 7 emotions, pick 1 correctly classified test example
    and produce a Grad-CAM overlay
  - Show a 2×7 grid: original face on top, Grad-CAM heatmap overlay below
  - Markdown: which facial regions activate per emotion (eyebrows for Angry,
    mouth for Happy, eye region for Fear). Connects to the spatial intuition
    from the EDA section 5

- [ ] Section 9 — Receptivity index simulation
  - Build a synthetic frame sequence: pick 200 test images in a controlled
    order that mimics a realistic sales call (e.g. starts neutral, climbs
    through happy/surprise, dips into fear, recovers)
  - For each frame: predict emotion → feed `ReceptivityIndex.update(...)`
  - Plot the receptivity index across the 200 frames
  - Annotate "best moments" (peaks) and "alert moments" (valleys)
  - Markdown: this is the qualitative validation that the index moves
    coherently with predicted emotions. Connects directly to the business
    case

- [ ] Section 10 — Final model selection
  - Markdown cell with weighted scoring:
    - Test accuracy (40%)
    - F1 macro (30%)
    - Inference latency (20%)
    - Model size (10%)
  - Compute a normalized score per criterion and a final weighted score per
    model. Show the table. State the winner explicitly with a 2-3 sentence
    justification.

- [ ] Section 11 — Critical reflection
  - Markdown only. Address:
    - Generalization from FER2013 (lab) to real webcam conditions
    - Limits of the emotion → commercial signal heuristic
    - Demographic bias in FER2013 and ethical implications for a commercial system
    - Whether 65-72% accuracy is acceptable for a coaching tool (vs deterministic decision-making)
    - Proposed future improvements (multimodal voice + face, AffectNet, TFLite deployment, fine-tuning on real sales call data)

- [ ] Section 12 — Summary
  - Standard closing markdown.

## Validation

- All plots render
- Grad-CAM produces visually meaningful heatmaps (faces visible underneath)
- Final selection is justified by the scoring table, not by gut feeling

## Output

Complete evaluation, final model chosen. Next phase: expose it via API.
