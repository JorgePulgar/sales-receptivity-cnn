from typing import Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf


class AugmentationPipeline:
    """cv2-based augmentation pipeline for (H, W, 1) float32 [0,1] images.

    Drop-in replacement for Keras ImageDataGenerator. Uses cv2 transforms
    directly to avoid the PIL conversion bug in TF 2.10 that zeroes out
    float32 [0,1] images when brightness_range is enabled.

    Exposes the same two methods used by the project:
      - random_transform(x): augment a single image in-place.
      - flow(x, y, ...): return a tf.keras.utils.Sequence for model.fit().
    """

    def __init__(
        self,
        rotation_range: float = 0,
        width_shift_range: float = 0.0,
        height_shift_range: float = 0.0,
        zoom_range: float = 0.0,
        horizontal_flip: bool = False,
        brightness_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.brightness_range = brightness_range

    def random_transform(self, x: np.ndarray) -> np.ndarray:
        """Apply random augmentation to a single image.

        Supports both grayscale (H, W, 1) and RGB (H, W, 3) inputs. The cv2
        transforms are channel-agnostic; we only normalise the shape going
        in and out so the rest of the pipeline does not care.

        Args:
            x: numpy array of shape (H, W, 1) grayscale or (H, W, 3) RGB,
                dtype float32, values in [0, 1].

        Returns:
            Augmented image with the same shape and dtype as ``x``,
            clipped to [0, 1].
        """
        has_singleton = x.ndim == 3 and x.shape[2] == 1
        img = x.squeeze(-1).copy() if has_singleton else x.copy()
        h, w = img.shape[:2]

        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), float(angle), 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        if self.width_shift_range > 0 or self.height_shift_range > 0:
            tx = np.random.uniform(-self.width_shift_range, self.width_shift_range) * w
            ty = np.random.uniform(-self.height_shift_range, self.height_shift_range) * h
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        if self.zoom_range > 0:
            scale = np.random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), 0.0, float(scale))
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        if self.horizontal_flip and np.random.random() > 0.5:
            img = img[:, ::-1]

        if self.brightness_range is not None:
            factor = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
            img = img * factor

        img = np.clip(img, 0.0, 1.0).astype(np.float32)
        if has_singleton:
            img = img[..., np.newaxis]
        return img

    def flow(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "_AugmentedSequence":
        """Return a Keras Sequence that yields augmented batches indefinitely.

        Compatible with model.fit() — implements __len__ and __getitem__.

        Args:
            x: Input images, shape (N, H, W, 1), float32 [0, 1].
            y: Labels array, shape (N, ...). Passed through unmodified.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle indices at the end of each epoch.
            seed: Optional random seed for reproducibility.
            sample_weight: Per-sample loss weights, shape (N,). When provided,
                each batch yields (x, y, w) so Keras applies weights correctly.
                Use this instead of model.fit(class_weight=...) on TF 2.10,
                which has a bug with one-hot labels and class_weight dicts.

        Returns:
            A tf.keras.utils.Sequence instance.
        """
        return _AugmentedSequence(self, x, y, batch_size, shuffle, seed, sample_weight)


class _AugmentedSequence(tf.keras.utils.Sequence):
    """Internal Sequence returned by AugmentationPipeline.flow()."""

    def __init__(
        self,
        pipeline: AugmentationPipeline,
        x: np.ndarray,
        y: Optional[np.ndarray],
        batch_size: int,
        shuffle: bool,
        seed: Optional[int],
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        self.pipeline = pipeline
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_weight = sample_weight
        self.indices = np.arange(len(x))
        if seed is not None:
            np.random.seed(seed)

    def __len__(self) -> int:
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx: int):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.stack([self.pipeline.random_transform(self.x[i]) for i in batch_idx])
        if self.y is not None:
            batch_y = self.y[batch_idx]
            if self.sample_weight is not None:
                return batch_x, batch_y, self.sample_weight[batch_idx]
            return batch_x, batch_y
        return batch_x

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)


def build_augmentation_pipeline(
    for_minority_classes: bool = False,
) -> AugmentationPipeline:
    """Build a cv2-based augmentation pipeline for training-time augmentation.

    Parameters mirror realistic webcam variation: small rotations (head tilt),
    slight shifts, mild zoom, brightness changes, and horizontal flip. Vertical
    flip is intentionally omitted — faces are not vertically symmetric.

    Args:
        for_minority_classes: If True, returns a stronger pipeline for
            under-represented classes (Disgust, Fear).

    Returns:
        A configured AugmentationPipeline instance.
    """
    if for_minority_classes:
        return AugmentationPipeline(
            rotation_range=25,
            width_shift_range=0.15,
            height_shift_range=0.15,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=(0.7, 1.3),
        )
    return AugmentationPipeline(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
    )
