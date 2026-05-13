from typing import Tuple

import tensorflow as tf
from tensorflow import keras


def build_mobilenet_head(
    input_shape: Tuple[int, int, int] = (64, 64, 3),
    num_classes: int = 7,
) -> tf.keras.Model:
    """Build MobileNetV2 with a frozen backbone and a trainable classification head.

    The MobileNetV2 base is loaded with ImageNet weights and fully frozen.
    Only the GlobalAveragePooling2D → Dense(256) → Dense(num_classes) head
    is trainable in this first stage. Fine-tuning of the backbone top layers
    is done separately via unfreeze_top_layers().

    Args:
        input_shape: (H, W, 3) — use (64, 64, 3) for FER2013 adapted to MobileNetV2.
        num_classes: Number of emotion classes.

    Returns:
        Compiled Keras model with base layers frozen.
    """
    base = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="mobilenet_ft")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_top_layers(
    model: tf.keras.Model,
    n_layers: int = 30,
) -> tf.keras.Model:
    """Unfreeze the top n layers of the MobileNetV2 backbone for fine-tuning.

    Freezes all backbone layers except the last n_layers, then recompiles with
    lr=1e-5 to avoid catastrophic forgetting of ImageNet features.

    Args:
        model: The model returned by build_mobilenet_head (base is model.layers[1]).
        n_layers: Number of layers from the top of the backbone to unfreeze.

    Returns:
        The same model (modified in place) after recompilation.
    """
    base = model.layers[1]  # MobileNetV2 is the second layer after InputLayer
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
