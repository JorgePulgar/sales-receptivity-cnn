# Phase 2 — Notebook 1: EDA (`notebooks/01_eda.ipynb`)

## Objective

Exploratory data analysis of FER2013. Understand the dataset and document
decisions that will shape preprocessing.

## Notebook structure

Follow the pedagogical format from `CLAUDE.md`: every code cell preceded by
a 2-5 sentence markdown cell explaining WHAT and WHY.

## Tasks

- [x] Section 0 — Introduction
  - Markdown cell: objective of the notebook, dataset used, what will be
    produced as output. Mention the connection with the business case (we need
    to know the data before deciding mapping and augmentation strategy).

- [x] Section 1 — Setup and data loading
  - Set random seeds
  - Import `src.data.loader.load_fer2013`
  - Load FER2013 and print shapes
  - Markdown explains why we use the existing FER2013 train/test split and
    carve a validation set from train

- [x] Section 2 — Class distribution
  - Bar plot of class counts in train / val / test
  - Table with absolute counts and percentages per class
  - Markdown discussion: which classes are minority (Disgust ~600 vs
    Happy ~8000), what this implies for training
  - Decision documented: use class weights, consider focused augmentation
    for Disgust and Fear

- [x] Section 3 — Sample grid per class
  - 8 random sample images per emotion (7×8 grid)
  - Markdown: intra-class variability, visual similarity between certain
    emotions (Fear/Surprise, Sad/Neutral, Angry/Disgust)
  - Anticipates which confusions we expect in the confusion matrix

- [x] Section 4 — Image quality analysis
  - Distribution of per-image pixel variance — detect near-uniform images
  - Distribution of mean pixel intensity — detect over/underexposed
  - Show 5 examples of the lowest-variance images
  - Markdown: are there enough anomalies to warrant cleaning? Document
    decision (most likely: keep as-is, FER2013 is a known benchmark with
    these quirks)

- [x] Section 5 — Mean intensity heatmap per class
  - For each class, compute mean image (average over all training images of
    that class)
  - Plot a 7-image grid of mean faces
  - Markdown: what spatial patterns are visible per emotion (mouth region
    for Happy, brow region for Angry). Connects to Grad-CAM expectations
    in Phase 6

- [ ] Section 6 — Visual similarity between classes
  - Compute mean image per class, then pairwise cosine similarity matrix
    between them
  - Heatmap of the 7×7 similarity matrix
  - Markdown: which pairs are most similar visually, which is consistent
    with the human-confusable emotions we anticipated

- [ ] Section 7 — Conclusions and decisions for preprocessing
  - Markdown cell summarizing:
    - Class imbalance → class weights + augmentation
    - No data cleaning needed (use FER2013 as-is)
    - Expected hard classes: Disgust, Fear
    - Expected confusions: Fear↔Surprise, Sad↔Neutral, Angry↔Disgust
    - Image size: keep 48×48 for CNN custom, resize to 64×64 for MobileNetV2

- [ ] Section 8 — Summary and link to next notebook
  - Standard closing markdown cell.

## Validation

- Notebook runs end-to-end without errors
- All plots render correctly
- Every code cell has a preceding markdown cell
- No business logic that should live in `src/` — only orchestration and analysis

## Output

EDA notebook complete. Decisions documented for Phase 3 (preprocessing).
