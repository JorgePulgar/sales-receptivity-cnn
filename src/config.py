from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
HISTORIES_DIR = MODELS_DIR / "histories"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# --- Labels ---
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Receptivity score per emotion (0–10 scale).
# High scores → buyer engaged; low scores → discomfort or disengagement.
EMOTION_TO_SCORE: dict[str, float] = {
    "happy": 9.0,
    "surprise": 7.0,
    "neutral": 5.0,
    "sad": 3.0,
    "fear": 2.0,
    "angry": 1.0,
    "disgust": 1.0,
}

# Human-readable commercial signal shown in the demo overlay.
EMOTION_TO_SIGNAL: dict[str, str] = {
    "happy": "Positive — reinforce proposal",
    "surprise": "Interested — elaborate on point",
    "neutral": "Attentive — continue normally",
    "sad": "Disengaged — check in with prospect",
    "fear": "Uncomfortable — slow down, clarify",
    "angry": "Resistant — pause and address concern",
    "disgust": "Resistant — pause and address concern",
}

# --- Hyperparameters ---
IMG_SIZE_CUSTOM = (48, 48)      # Native FER2013 resolution
IMG_SIZE_MOBILENET = (96, 96)   # MobileNetV2's official minimum useful size — 64×64 collapses the backbone to a 2×2 feature map
BATCH_SIZE = 64
RANDOM_SEED = 42
