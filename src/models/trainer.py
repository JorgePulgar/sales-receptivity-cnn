import json
from pathlib import Path
from typing import Any, List, Tuple, Union

import tensorflow as tf
from tensorflow import keras


def get_default_callbacks(
    model_name: str,
    models_dir: Path,
) -> List[keras.callbacks.Callback]:
    """Return the standard set of training callbacks.

    Includes EarlyStopping (patience=12, restores best weights),
    ReduceLROnPlateau (patience=5, factor=0.5), and ModelCheckpoint
    saving the epoch with the lowest validation loss. Patience is wide
    because FER2013 training spends ~6-8 epochs in a majority-class
    plateau before escaping; shorter patience kills training during that
    exact window.

    Args:
        model_name: Used to name the checkpoint file (e.g. 'cnn_custom').
        models_dir: Directory where the .keras checkpoint will be written.

    Returns:
        List of three configured Keras callbacks.
    """
    checkpoint_path = models_dir / f"{model_name}_best.keras"
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=5,
            factor=0.5,
            min_lr=1e-7,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]


def train_model(
    model: tf.keras.Model,
    train_data: Union[Tuple[Any, Any], keras.utils.Sequence],
    val_data: Tuple[Any, Any],
    class_weights: dict,
    epochs: int,
    callbacks: List[keras.callbacks.Callback],
    batch_size: int,
) -> keras.callbacks.History:
    """Fit the model and return the training history.

    Args:
        model: Compiled Keras model.
        train_data: Either a (X_train, y_train) tuple or a Keras data generator.
        val_data: (X_val, y_val) tuple.
        class_weights: Dict mapping integer class index to its loss weight.
        epochs: Maximum number of training epochs (EarlyStopping may fire earlier).
        callbacks: List of Keras callbacks to attach.
        batch_size: Mini-batch size; used only when train_data is a tuple.

    Returns:
        Keras History object containing per-epoch metric values.
    """
    if isinstance(train_data, tuple):
        X_train, y_train = train_data
        return model.fit(
            X_train,
            y_train,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
        )
    return model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )


def save_history(history: keras.callbacks.History, path: Path) -> None:
    """Serialize a Keras History object to a JSON file.

    Converts all metric values to plain floats so the file is JSON-safe and
    can be reloaded without TensorFlow to regenerate training plots.

    Args:
        history: The History object returned by model.fit().
        path: Destination .json file path (parent directories are created if needed).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
