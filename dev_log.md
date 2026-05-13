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
