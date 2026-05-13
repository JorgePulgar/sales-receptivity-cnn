from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from src import config


def load_fer2013(
    data_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load FER2013 from the Kaggle folder layout and split into train/val/test.

    Reads train/<emotion>/*.jpg and test/<emotion>/*.jpg. Carves a stratified
    15% validation set from the Kaggle train split; the Kaggle test split is
    used as-is for final evaluation.

    Args:
        data_dir: Path to the raw data directory containing 'train' and 'test' subdirs.

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test) where images are
        uint8 grayscale arrays of shape (n, 48, 48) and labels are int32
        indices 0–6 following the order in config.EMOTION_LABELS.
    """
    label_map = {label: idx for idx, label in enumerate(config.EMOTION_LABELS)}

    def _read_split(split_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        images, labels = [], []
        for emotion_dir in sorted(split_dir.iterdir()):
            if not emotion_dir.is_dir():
                continue
            label_idx = label_map[emotion_dir.name.lower()]
            for img_path in sorted(emotion_dir.iterdir()):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                images.append(img)
                labels.append(label_idx)
        return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int32)

    X_all, y_all = _read_split(data_dir / "train")
    X_test, y_test = _read_split(data_dir / "test")

    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=0.15,
        stratify=y_all,
        random_state=config.RANDOM_SEED,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def to_rgb(images: np.ndarray) -> np.ndarray:
    """Convert grayscale images to 3-channel RGB by duplicating the channel.

    Args:
        images: Array of shape (n, H, W) or (n, H, W, 1).

    Returns:
        Array of shape (n, H, W, 3) with the same dtype.
    """
    if images.ndim == 3:
        images = images[..., np.newaxis]
    return np.repeat(images, 3, axis=-1)


def resize_batch(images: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize a batch of images to target_size using bilinear interpolation.

    Args:
        images: Array of shape (n, H, W) or (n, H, W, C).
        target_size: Desired output (height, width).

    Returns:
        Array with shape (n, height, width) or (n, height, width, C),
        same dtype as input.
    """
    has_channel = images.ndim == 4
    h, w = target_size
    resized = [cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR) for img in images]
    result = np.array(resized, dtype=images.dtype)
    if has_channel and result.ndim == 3:
        result = result[..., np.newaxis]
    return result
