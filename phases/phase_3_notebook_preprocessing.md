# Phase 3 — Notebook 2: Preprocessing (`notebooks/02_preprocessing.ipynb`)

## Objective

Transform raw FER2013 arrays into training-ready tensors. Compute class
weights. Visualize augmentation effects.

## Tasks

- [x] Section 0 — Introduction
  - Markdown: objective. Mention that we will produce two versions of the data
    (grayscale 48×48 for the custom CNN, RGB 64×64 for MobileNetV2) and save
    them to `data/processed/` for fast reload in the training notebook.

- [x] Section 1 — Setup and data loading
  - Set seeds
  - Load FER2013 via `src.data.loader.load_fer2013`
  - Print shapes again to confirm starting point

- [x] Section 2 — Normalization
  - Convert uint8 [0, 255] → float32 [0, 1] by dividing by 255
  - Markdown: why normalization is needed (training stability, faster
    convergence, also required for MobileNetV2 input expectations)

- [x] Section 3 — Reshape for the custom CNN
  - Add channel dimension: `(n, 48, 48) → (n, 48, 48, 1)`
  - Markdown: Keras Conv2D requires explicit channel axis

- [x] Section 4 — Prepare MobileNetV2 inputs
  - Use `src.data.loader.resize_batch` to resize to 64×64
  - Use `src.data.loader.to_rgb` to duplicate channels
  - Result shape: `(n, 64, 64, 3)`
  - Markdown: MobileNetV2 was pretrained on ImageNet which expects RGB inputs.
    Channel duplication is a standard, cheap workaround for grayscale datasets.
    Mention that resizing to 64×64 is a compromise between 48×48 (native FER2013)
    and 224×224 (ImageNet native), balancing accuracy vs latency

- [x] Section 5 — One-hot encoding of labels
  - Convert integer labels (0–6) to one-hot vectors of length 7
  - Use `tf.keras.utils.to_categorical`
  - Markdown: needed because we use `categorical_crossentropy` (alternative
    would be `sparse_categorical_crossentropy` with integer labels — mention
    the discarded option)

- [x] Section 6 — Class weights computation
  - Use `sklearn.utils.class_weight.compute_class_weight('balanced', ...)`
    on the integer training labels
  - Build a `dict` mapping class index → weight
  - Print the resulting weights per emotion name
  - Markdown: how class weights modify the loss to compensate imbalance.
    Why this is preferred over aggressive oversampling for FER2013 (avoids
    duplicating low-quality minority samples)

- [ ] Section 7 — Augmentation pipeline configuration
  - Import `src.data.augmentation.build_augmentation_pipeline`
  - Build the default generator
  - Markdown: parameter justification — rotations within head-tilt range,
    no vertical flip (faces are not vertically symmetric), brightness
    variation to simulate different lighting in real webcam usage

- [ ] Section 8 — Augmentation visualization
  - Pick 1 random training image
  - Apply the augmentation pipeline 8 times
  - Show original + 8 augmented variants in a 3×3 grid
  - Markdown: what the model will see during training; this gives intuition
    for why augmentation helps generalization to webcam conditions

- [ ] Section 9 — Save processed datasets
  - Save `data/processed/fer2013_gray.npz` with X_train, y_train, X_val,
    y_val, X_test, y_test (48×48×1, float32, one-hot labels)
  - Save `data/processed/fer2013_rgb64.npz` with the same splits at 64×64×3
  - Save `data/processed/class_weights.json` with the class weights dict
  - Markdown: why we persist the processed data (notebook 3 can reload in
    seconds instead of reprocessing)

- [ ] Section 10 — Summary and link to next notebook
  - Standard closing markdown.

## Validation

- Both `.npz` files exist and reload correctly
- `class_weights.json` exists and has 7 entries
- Augmentation grid renders correctly

## Output

Training-ready tensors and class weights. Next phase: inference modules
(can be built in parallel with training, both depend only on Phase 1).
