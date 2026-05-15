from typing import Tuple

import tensorflow as tf
from tensorflow import keras


def build_mobilenet_head(
    input_shape: Tuple[int, int, int] = (96, 96, 3),
    num_classes: int = 7,
) -> tf.keras.Model:
    """Build MobileNetV2 with a frozen backbone and a trainable classification head.

    The MobileNetV2 base is loaded with ImageNet weights and fully frozen.
    Only the GlobalAveragePooling2D → Dense(256) → Dense(num_classes) head
    is trainable in this first stage. Fine-tuning of the backbone top layers
    is done separately via unfreeze_top_layers().

    Input is 96×96, MobileNetV2's official minimum useful size. At 64×64 the
    backbone collapses to a 2×2 spatial feature map after ~32× downsampling,
    leaving GlobalAveragePooling almost nothing to average; at 96×96 it
    stays at 3×3, which retains enough spatial information for the head to
    discriminate emotions. Label smoothing (0.1) matches the custom CNN.

    Args:
        input_shape: (H, W, 3) — use (96, 96, 3) for FER2013 adapted to MobileNetV2.
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
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="mobilenet_ft")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    return model


def unfreeze_top_layers(
    model: tf.keras.Model,
    n_layers: int = 60,
) -> tf.keras.Model:
    """Unfreeze the top n layers of the MobileNetV2 backbone for fine-tuning.

    Freezes all backbone layers except the last n_layers, then recompiles with
    lr=5e-5. Unfreezing 60 of MobileNetV2's 154 layers (~40 %) gives the
    optimiser enough capacity to adapt the spatial-feature detectors to faces;
    the slightly higher LR (5e-5 vs the conventional 1e-5) is safe because
    we are starting from a head that already reached ~50 % val_acc on a
    96×96 input — small gradients would barely move the backbone.

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
        optimizer=keras.optimizers.Adam(learning_rate=5e-5),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    return model
