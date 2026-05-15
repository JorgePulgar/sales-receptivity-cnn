from typing import Tuple

import tensorflow as tf
from tensorflow import keras


def build_cnn_custom(
    input_shape: Tuple[int, int, int] = (48, 48, 1),
    num_classes: int = 7,
) -> tf.keras.Model:
    """Build and compile the custom 4-block CNN for FER2013 classification.

    Architecture: four Conv2D blocks with 32→64→128→256 filters, each followed
    by BatchNorm, ReLU, MaxPool(2), and Dropout(0.1). Two dense layers
    (512, 256) with Dropout(0.3) before the softmax output head.

    Dropout is intentionally light (0.1 conv / 0.3 dense). Heavier dropout
    (0.25/0.5) combined with augmentation locks the optimiser into a
    Happy-only minimum on FER2013: total regularisation noise exceeds the
    early learning signal, the model never escapes the majority-class
    plateau, and val_accuracy freezes at ≈ 0.25 for all 50 epochs.

    Args:
        input_shape: (H, W, C) — use (48, 48, 1) for grayscale FER2013.
        num_classes: Number of emotion classes (7 for FER2013).

    Returns:
        Compiled Keras model ready for training.
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs

    for filters in (32, 64, 128, 256):
        x = keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.MaxPooling2D(2)(x)
        x = keras.layers.Dropout(0.1)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="cnn_custom")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    return model
