# Phase 5 — Notebook 3: Training (`notebooks/03_model_training.ipynb`)

## Objective

Train two architectures (CNN custom and MobileNetV2 fine-tuned). Save
trained models and training histories.

## Tasks

- [x] Section 0 — Introduction
  - Markdown: objective. Mention that we train two models for comparison
    and that all evaluation happens in Notebook 4. Verify GPU here.

- [ ] Section 1 — Setup
  - Set all seeds (numpy, tensorflow, random)
  - Verify GPU with `tf.config.list_physical_devices('GPU')` and print
    result. If no GPU, print a warning but continue.
  - Imports

- [ ] Section 2 — Load preprocessed data
  - Load `data/processed/fer2013_gray.npz`
  - Load `data/processed/fer2013_rgb64.npz`
  - Load `data/processed/class_weights.json`
  - Print shapes
  - Markdown: why we reload from `.npz` instead of redoing preprocessing
    (reproducibility + speed)

- [ ] Section 3.1 — Build custom CNN (Model A)
  - `build_cnn_custom(input_shape=(48, 48, 1))`
  - `model.summary()`
  - Markdown: brief recap of the architecture choice (4 conv blocks balance
    capacity vs overfitting on 48×48 images)

- [ ] Section 3.2 — Configure callbacks for custom CNN
  - Use `get_default_callbacks('cnn_custom', MODELS_DIR)`
  - Markdown: callbacks explained briefly (early stopping prevents overfitting,
    LR reduction helps fine-grained convergence)

- [ ] Section 3.3 — Train custom CNN
  - 50 epochs max, batch size 64
  - Pass `class_weight=class_weights`
  - Capture `history`
  - Markdown: expected behavior — accuracy should plateau around 60-68%,
    early stopping likely fires before epoch 50

- [ ] Section 3.4 — Save custom CNN
  - Save model to `models/cnn_custom.keras`
  - Save `history.history` to `models/histories/cnn_custom_history.json`
  - Markdown: why we save both (model for inference, history for plotting
    in Notebook 4 without retraining)

- [ ] Section 4.1 — Build MobileNetV2 head (Model B)
  - `build_mobilenet_head(input_shape=(64, 64, 3))`
  - `model.summary()` and count trainable vs non-trainable params
  - Markdown: only the top dense layers train in this stage; the MobileNetV2
    backbone is frozen with ImageNet weights

- [ ] Section 4.2 — Stage 1: train the MobileNetV2 head
  - 10 epochs, batch size 64, lr=1e-3 (already set by builder)
  - Class weights
  - Markdown: stage 1 lets the head adapt to FER2013 before disturbing
    the pretrained features

- [ ] Section 4.3 — Stage 2: unfreeze top layers and fine-tune
  - `unfreeze_top_layers(model, n_layers=30)`
  - `model.summary()` again — trainable params should increase
  - Train 20 more epochs with the same callbacks (now with lr=1e-5 set by
    `unfreeze_top_layers`)
  - Markdown: fine-tuning the top layers adapts pretrained features to the
    emotion domain. Very low LR prevents catastrophic forgetting

- [ ] Section 4.4 — Save MobileNetV2
  - Save final model to `models/mobilenet_ft.keras`
  - Save the combined history (stage 1 + stage 2) to
    `models/histories/mobilenet_ft_history.json`
  - Markdown: history concatenation logic explained

- [ ] Section 5 — Quick sanity check
  - For each model, predict on the first 16 test images
  - Print predicted vs true labels
  - Markdown: just a smoke test — full evaluation comes in Notebook 4

- [ ] Section 6 — Summary and link to next notebook
  - Standard closing markdown. Mention training times observed and any issues
    encountered.

## Validation

- Both `.keras` files exist in `models/`
- Both history JSON files exist in `models/histories/`
- Notebook ran without OOM errors (if it did, reduce batch size in `config.py`)

## Notes

- If `.keras` saving fails on TF 2.10, fall back to `.h5` and update
  filenames accordingly. Document the fallback in a markdown cell.
- Training time on a modern NVIDIA GPU (e.g. RTX 3060 or better): about
  3-8 min for CNN custom, 8-20 min for MobileNetV2 (both stages combined).
- Do NOT evaluate on the test set here beyond the smoke check — keep
  test-set analysis isolated in Notebook 4.

## Output

Two trained models and two histories on disk. Next phase: evaluation.
