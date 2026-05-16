"""Python-side smoke test: predict on a 48x48 gray image (all pixels=128).
Compare the printed softmax vector against the browser output in smoke.html.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.inference.emotion_classifier import EmotionClassifier
from src import config

clf = EmotionClassifier(
    Path("models/cnn_custom.keras"),
    input_size=(48, 48),
    use_rgb=False,
)
face = np.full((48, 48), 128, dtype=np.uint8)
label, conf, probs = clf.predict(face)

print("Softmax probabilities:")
for lbl, p in zip(config.EMOTION_LABELS, probs):
    print(f"  {lbl:<8} {p:.6f}")
print(f"\nPredicted: {label}  ({conf*100:.2f}%)")
