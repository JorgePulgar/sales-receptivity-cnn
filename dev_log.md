# Development Log

Problems and decisions encountered during development, for reference when writing the final README.

---

## Phase 0 — Environment setup

### Problem: TensorFlow 2.10.1 incompatible with Python 3.14

**What happened:** The development machine had Python 3.14.3 as the system interpreter.
Running `pip install -r requirements.txt` failed immediately:

```
ERROR: Could not find a version that satisfies the requirement tensorflow==2.10.1
ERROR: No matching distribution found for tensorflow==2.10.1
```

**Root cause:** TensorFlow 2.10.1 only publishes wheels for Python 3.7–3.10.
Python 3.14 is too new — no matching wheel exists on PyPI for any platform.

**Why we can't just upgrade TF:** TensorFlow 2.10 is the last release with native
CUDA GPU support on Windows. Versions 2.11+ dropped native Windows GPU support.
Upgrading TF would mean losing GPU acceleration on both Windows machines in this project.

**Solution:** Install Python 3.10 via Miniconda (isolated, does not affect system Python):

```cmd
winget install Anaconda.Miniconda3
conda create -n sales-cnn python=3.10 -y
conda activate sales-cnn
pip install -r requirements.txt
```

---

### Problem: streamlit 1.31.1 conflicts with TF 2.10.1 via protobuf

**What happened:** `pip install -r requirements.txt` failed with a dependency conflict:

```
tensorflow 2.10.1 depends on protobuf<3.20
streamlit 1.31.1 depends on protobuf>=3.20
```

These are mutually exclusive — no single protobuf version satisfies both.

**Solution:** Downgraded streamlit to `1.18.1`, the last version before the
`protobuf>=3.20` requirement was introduced. No functional difference for this project.

---

### Decision: GPU work deferred to a separate machine

**Context:** The development machine has no GPU. The GPU (NVIDIA, Windows) is on a
separate PC. Training on FER2013 without a GPU would take impractically long.

**Decision:** All phases except Phase 5 (model training) are developed and tested on
the CPU machine. When Phase 5 is reached, the repository is pulled on the GPU machine,
the same Miniconda environment is set up there, and training is run. The resulting
`models/histories/*.json` files are committed back and used for evaluation plots in
Phase 6 without retraining.

**Why this works:** The project is structured so that `src/` modules have no side
effects on import and notebooks are self-contained. Switching machines mid-project
requires only a `git pull` and `conda activate`.

---

## Phase 5 — Model training

Three bugs piled on top of each other in `notebooks/03_model_training.ipynb`. Each one
individually freezes training; together they made the failure mode unstable across runs
and the diagnosis non-obvious. Documenting all three because the *interaction* is what
made this hard to debug.

### Problem 1: `class_weight` collapses training on TF 2.10 + one-hot labels

**What happened:** First training run of the custom CNN. Loss froze at exactly
`log(7) ≈ 1.946` from epoch 2 onward. val_accuracy oscillated randomly near 1/7 ≈ 14 %.
Lowering the LR (manually and via `ReduceLROnPlateau`) had zero effect. The model was
producing uniform 1/7 probabilities for every input.

**Root cause:** In TF 2.10, passing a `class_weight` dict to `model.fit()` when the
labels are one-hot encoded (categorical_crossentropy) corrupts the weighted-loss path.
The implementation was designed for integer sparse labels; with one-hot it creates a
loss landscape where uniform output is a stable minimum. Adam lands there in epoch 1
and there are no gradients pushing it out.

**Solution:** Pass `class_weight=None` to `model.fit()`. The class_weights dict is still
computed in Notebook 2 and capped at 3× in Notebook 3 for didactic continuity, but it is
never handed to Keras.

### Problem 2: `sample_weight` through a `Sequence` has the same bug

**What happened:** First attempted workaround for Problem 1 was to convert the class
weights into per-sample weights and pass them through the augmentation generator's
`flow(sample_weight=...)`, so the `_AugmentedSequence.__getitem__` returns `(x, y, w)`
triples. Training collapsed *again*, this time at a loss of ≈ 1.72 (= `log(7) × mean
sample weight ≈ 1.946 × 0.886`). Same uniform-output failure, slightly disguised by the
weights.

**Root cause:** TF 2.10's weighted-loss path is broken for one-hot labels regardless of
whether the weights arrive via `class_weight` or `sample_weight`. The fact that the
weights come from a `Sequence` doesn't help — they are applied through the same code
path.

**Solution:** Train without any per-class or per-sample weighting. The class imbalance
on FER2013 (Happy 25 % vs Disgust 1.5 %) is real but small enough to handle with the
other fixes below. If real weighting is ever needed, options that don't trigger this
bug: pre-resample the training set (duplicate minority samples), switch to sparse
integer labels with `sparse_categorical_crossentropy`, or write a custom weighted loss.

**Diagnostic signature for both Problems 1 and 2:** training loss flat at
`log(num_classes) × mean(weights)`, val_accuracy locked at exactly `class_freq[majority]`
(here 0.2512 = Happy's share of the val set) or oscillating near `1/num_classes`. The
exact loss value differs by which weighting is applied; the *flatness* is the tell.

### Problem 3: heavy dropout + augmentation locks the optimiser in *Happy* minimum

**What happened:** After removing all weighting (Problems 1 and 2 fixed), the model
**still** could not learn with augmentation enabled. val_accuracy stayed at exactly
0.2512 — i.e. predicting *Happy* for every validation sample — for all 50 epochs.
val_loss did drop slowly (1.81 → 1.73), meaning the model was becoming more confident
on *Happy* without ever broadening its predictions. Disabling augmentation, with
everything else identical, recovered learning: the model escaped the *Happy*-only
plateau at epoch ~7 and reached 48.7 % val_accuracy at epoch 50.

**Root cause:** The original CNN used Dropout(0.25) after each of four conv blocks and
Dropout(0.5) after each of two dense layers. Combined with BatchNorm, Adam at
lr = 1e-3, and the augmentation pipeline's added noise (random rotation, shifts, zoom,
brightness, flip), the *total* regularisation signal exceeds the supervised signal in
the early epochs. The optimiser cannot find any direction better than "predict the
majority class with growing confidence", and the entire ReduceLROnPlateau schedule fires
while still inside that minimum, killing what little gradient remained.

**Solution:** Reduce dropout from 0.25 / 0.5 to **0.1 / 0.3** in `src/models/cnn_custom.py`.
With lighter regularisation the model escapes the *Happy* plateau by epoch 2-3 and
reaches **60.9 % best val_accuracy** with augmentation fully enabled — well inside the
58-65 % target band documented for this architecture on FER2013. Train/val gap stays
near zero, so overfitting has not been re-introduced.

### Problem 4: augmentation pipeline assumed grayscale, broke on RGB inputs

**What happened:** Stage 1 of the MobileNetV2 training failed at the first batch with:

```
ValueError: too many values to unpack (expected 2)
  File src/data/augmentation.py, line 46, in random_transform
    h, w = img.shape
```

**Root cause:** `random_transform` was written when only the custom CNN existed and
inputs were always `(48, 48, 1)` grayscale. The line `img = x.squeeze().copy()`
collapses the singleton channel for grayscale, yielding `(48, 48)`, and `h, w = img.shape`
unpacks cleanly. When MobileNetV2 Stage 1 reused `build_augmentation_pipeline()` on its
`(64, 64, 3)` RGB inputs, `squeeze()` did nothing (no singleton dims) and `img.shape`
remained `(64, 64, 3)` — three values, two-variable unpack, hard failure.

**Solution:** Generalise `random_transform` to handle both grayscale and RGB. Track
whether the input had a singleton channel; squeeze it only in that case; use
`img.shape[:2]` for the spatial dims so the unpack works regardless of channel count;
re-attach the singleton at the end only for grayscale. cv2 transforms (`warpAffine`,
flip, brightness multiply) are channel-agnostic, so the body of the function did not
need to change.

**Why this was easy to miss:** the function had a working unit test path via the custom
CNN, and the docstring explicitly said `(H, W, 1)`. The MobileNet code reused the same
pipeline factory without noticing the shape assumption.

### Supporting change: wider callback patience

`get_default_callbacks` in `src/models/trainer.py` previously used `EarlyStopping(patience=7)`
and `ReduceLROnPlateau(patience=3)`. With the dropout fix these patiences are no longer
strictly necessary, but during the investigation it became clear that the original
values would kill or starve training during *any* short plateau (e.g., the 2-3 epoch
*Happy* warm-up phase that still exists). Bumped to **patience=12** (EarlyStopping) and
**patience=5** (ReduceLROnPlateau) so the callbacks don't pre-empt a model that just
needs a few epochs to stabilise.

### Round 2 — pushing both models past the baseline

After the three bugs above were resolved, the custom CNN converged at ~58 % val_acc
and the MobileNetV2 (Stage 1 + Stage 2) at ~45 %. Below the 58–65 % target for the
custom CNN, and well below "transfer learning beats from-scratch" for the MobileNet.
A second pass of targeted fixes was applied to push both models higher.

**Custom CNN:**
- **Label smoothing 0.1** added to `categorical_crossentropy`. FER2013 contains
  genuinely ambiguous expressions (Sad/Neutral, Angry/Disgust) where a one-hot
  target is wrong even for a perfect classifier. Smoothing softens the targets so
  the optimiser does not waste capacity becoming over-confident on those edge cases.
  Typical gain: +0.5–1 pt val_acc.
- **Training extended from 50 → 75 epochs.** The 50-epoch run was still climbing
  on val_loss at the cap, so EarlyStopping (patience=12) is now the real stop
  signal rather than an arbitrary epoch number.

**MobileNetV2** (the bigger fix):
- **Input resolution raised from 64×64 to 96×96.** This is the central change.
  MobileNetV2 has ~32× internal downsampling, so a 64×64 input produces a 2×2
  spatial feature map at the output of the backbone — almost no information for
  `GlobalAveragePooling2D` to summarise, and ImageNet-pretrained filters tuned
  for 224×224 receptive fields cannot do useful work. 96×96 is documented by
  Keras as MobileNetV2's official minimum useful size; it yields a 3×3 feature
  map and recovers most of the lost capacity.
- **Head dropout reduced 0.5 → 0.3** — same augmentation-vs-regularisation
  trade-off lesson the custom CNN taught us.
- **Stage 1 extended 10 → 20 epochs** so the head fully converges before
  unfreezing the backbone.
- **Stage 2 expanded: 30 → 60 unfrozen layers, LR 1e-5 → 5e-5, 20 → 30 epochs.**
  ImageNet features for everyday objects are a poor prior for FER2013 faces, so
  more of the backbone is worth re-tuning. The slightly higher LR is safe because
  Stage 1 already left the head well-conditioned.

**Required cascade through the codebase:** `config.IMG_SIZE_MOBILENET` changed
from `(64, 64)` to `(96, 96)`; preprocessing (`02_preprocessing.ipynb`) now
saves `data/processed/fer2013_rgb96.npz` instead of `fer2013_rgb64.npz`;
training (`03_model_training.ipynb`) and evaluation (`04_evaluation.ipynb`)
load the new file. The FastAPI service and Streamlit demo pick up the new
resolution automatically through `config.IMG_SIZE_MOBILENET`.

### Lessons for the README

- TF 2.10's weighted-loss path is unsafe with one-hot labels — assume both
  `class_weight` and `sample_weight` are broken and design around it.
- When debugging a stuck CNN, compute `log(num_classes) × mean(weights)` and check
  whether your loss is sitting at that number. If yes, it is in the uniform-output
  minimum and adjusting LR or epochs will not help.
- Augmentation and dropout *both* add regularisation noise. When using one, dial back
  the other. The conventional 0.25 / 0.5 dropout values assume no input augmentation.
- A flat val_accuracy exactly equal to the majority-class frequency is the signature of
  majority-class collapse, not bad luck. Treat it as a hard-fail signal, not a "needs
  more epochs" signal.
- Pretrained backbones are not unconditionally better than from-scratch CNNs.
  MobileNetV2 underperformed the custom CNN at 64×64 input because its filters
  were tuned for 224×224 and the internal downsampling left almost no spatial
  feature map. Always check the backbone output spatial dimensions before
  trusting transfer learning — if it is ≤ 3×3, raise the input resolution or
  pick a different backbone.
- Label smoothing is cheap insurance on datasets with genuinely ambiguous labels
  (FER2013 Sad/Neutral, Angry/Disgust). One line in `model.compile`, +0.5–1 pt
  val_acc, basically no risk.
- "Still climbing at the epoch cap" is not a healthy stopping condition. Either
  raise the cap and let EarlyStopping decide, or fix the actual learning-rate
  schedule. Don't ship a model that the optimiser was still improving.
