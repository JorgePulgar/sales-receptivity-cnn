from pathlib import Path
from typing import List, Tuple

import numpy as np

from src import config


class EmotionClassifier:
    """Loads a trained Keras model and runs emotion inference on face ROIs.

    Preprocessing (normalisation, channel duplication, batch axis) is handled
    internally so callers only need to pass a cropped face array.
    """

    def __init__(
        self,
        model_path: Path,
        input_size: Tuple[int, int],
        use_rgb: bool,
        labels: List[str] = config.EMOTION_LABELS,
    ) -> None:
        """Load the Keras model from disk.

        Args:
            model_path: Path to the ``.keras`` or ``.h5`` model file.
            input_size: ``(height, width)`` expected by the model.
            use_rgb: If ``True``, duplicate grayscale channel to RGB before
                inference (required for MobileNetV2-based models).
            labels: Ordered list of emotion class names. Defaults to
                ``config.EMOTION_LABELS``.

        Raises:
            FileNotFoundError: If ``model_path`` does not exist.
        """
        import tensorflow as tf  # deferred so the module is GPU-free at import

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self._model = tf.keras.models.load_model(str(model_path))
        self._input_size = input_size
        self._use_rgb = use_rgb
        self._labels = labels

    def predict(
        self, face_roi: np.ndarray
    ) -> Tuple[str, float, np.ndarray]:
        """Run inference on a single face crop.

        Args:
            face_roi: Grayscale or single-channel face crop of any size.
                Will be resized internally to ``self._input_size``.

        Returns:
            Tuple of:
            - ``emotion_label``: predicted class name (str)
            - ``confidence``: probability of the predicted class (float 0–1)
            - ``probabilities``: full softmax vector (np.ndarray, shape (7,))
        """
        tensor = self._preprocess(face_roi)           # (1, H, W, C)
        probs = self._model.predict(tensor, verbose=0)[0]  # (7,)
        idx = int(np.argmax(probs))
        return self._labels[idx], float(probs[idx]), probs.astype(np.float32)

    def predict_batch(
        self, face_rois: np.ndarray
    ) -> List[Tuple[str, float, np.ndarray]]:
        """Run inference on a batch of face crops for video processing.

        Args:
            face_rois: Array of shape ``(n, H, W)`` or ``(n, H, W, 1)``.
                All crops are resized to ``self._input_size`` internally.

        Returns:
            List of ``(emotion_label, confidence, probabilities)`` tuples,
            one per input crop.
        """
        import cv2

        tensors = np.stack(
            [self._preprocess(roi)[0] for roi in face_rois], axis=0
        )  # (n, H, W, C)
        batch_probs = self._model.predict(tensors, verbose=0)  # (n, 7)
        results = []
        for probs in batch_probs:
            idx = int(np.argmax(probs))
            results.append((self._labels[idx], float(probs[idx]), probs.astype(np.float32)))
        return results

    # ------------------------------------------------------------------
    def _preprocess(self, face_roi: np.ndarray) -> np.ndarray:
        """Normalise, resize, add channel and batch axes."""
        import cv2

        h, w = self._input_size
        # Ensure 2-D before resize
        if face_roi.ndim == 3 and face_roi.shape[2] == 1:
            face_roi = face_roi[:, :, 0]
        resized = cv2.resize(face_roi, (w, h), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        arr = arr[..., np.newaxis]          # (H, W, 1)
        if self._use_rgb:
            arr = np.repeat(arr, 3, axis=-1)  # (H, W, 3)
        return arr[np.newaxis]              # (1, H, W, C)
