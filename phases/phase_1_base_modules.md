# Phase 1 — Base modules in `src/data` and `src/models`

## Objective

Clean, reusable modules built BEFORE touching notebooks. Notebooks will
import these, never duplicate logic.

## Tasks

### `src/data/loader.py`

- `load_fer2013(data_dir: Path) -> Tuple[np.ndarray, ...]`
  - Reads FER2013 from `data/raw/`
  - FER2013 from Kaggle (`msambare/fer2013`) comes as folder structure
    `train/<emotion>/*.jpg` and `test/<emotion>/*.jpg`
  - Returns `(X_train, y_train, X_val, y_val, X_test, y_test)` as numpy arrays
  - Split: take the original train, do stratified 70/15 split into
    train/val. Use the original test as test.
  - `random_state=42`
  - Images returned as `(n, 48, 48)` uint8 grayscale
  - Labels returned as integer class indices (0–6) following the order in
    `config.EMOTION_LABELS`
- `to_rgb(images: np.ndarray) -> np.ndarray`
  - Converts `(n, H, W)` or `(n, H, W, 1)` to `(n, H, W, 3)` by channel duplication
- `resize_batch(images: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray`
  - Uses `cv2.resize` per image

### `src/data/augmentation.py`

- `build_augmentation_pipeline()` returning a Keras 2.10
  `ImageDataGenerator` with:
  - `rotation_range=15`
  - `width_shift_range=0.1`
  - `height_shift_range=0.1`
  - `zoom_range=0.1`
  - `horizontal_flip=True`
  - `brightness_range=(0.8, 1.2)`
  - `fill_mode='nearest'`
- Parameter `for_minority_classes: bool = False` that, if True, returns a
  stronger pipeline (rotation 25, zoom 0.2, etc.) for focused augmentation

### `src/models/cnn_custom.py`

- `build_cnn_custom(input_shape: Tuple[int, int, int] = (48, 48, 1),
   num_classes: int = 7) -> tf.keras.Model`
- Architecture:
  ```
  Input
  → Conv2D(32, 3) + BN + ReLU + MaxPool(2) + Dropout(0.25)
  → Conv2D(64, 3) + BN + ReLU + MaxPool(2) + Dropout(0.25)
  → Conv2D(128, 3) + BN + ReLU + MaxPool(2) + Dropout(0.25)
  → Conv2D(256, 3) + BN + ReLU + MaxPool(2) + Dropout(0.25)
  → Flatten
  → Dense(512) + ReLU + Dropout(0.5)
  → Dense(256) + ReLU + Dropout(0.5)
  → Dense(num_classes) + Softmax
  ```
- Compiles with Adam(lr=1e-3), `categorical_crossentropy`, `accuracy` metric

### `src/models/mobilenet_finetune.py`

- `build_mobilenet_head(input_shape: Tuple[int, int, int] = (64, 64, 3),
   num_classes: int = 7) -> tf.keras.Model`
  - MobileNetV2 with `weights='imagenet'`, `include_top=False`, base frozen
  - GlobalAveragePooling2D → Dense(256) + ReLU + Dropout(0.5) → Dense(num_classes) + Softmax
  - Compiles with Adam(lr=1e-3)
- `unfreeze_top_layers(model: tf.keras.Model, n_layers: int = 30) -> tf.keras.Model`
  - Unfreezes the top `n_layers` of the MobileNetV2 base
  - Recompiles with Adam(lr=1e-5)
  - Returns the same model (modified in place)

### `src/models/trainer.py`

- `get_default_callbacks(model_name: str, models_dir: Path) -> List[Callback]`
  - `EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)`
  - `ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-7)`
  - `ModelCheckpoint(models_dir / f'{model_name}_best.keras', save_best_only=True)`
- `train_model(model, train_data, val_data, class_weights, epochs, callbacks, batch_size) -> History`
  - Returns the Keras `History` object
- `save_history(history, path: Path) -> None`
  - Serializes `history.history` to JSON

## Validation

- `from src.data.loader import load_fer2013; X_train, y_train, X_val, y_val, X_test, y_test = load_fer2013(...)`
  works and prints shapes consistent with FER2013
- `from src.models.cnn_custom import build_cnn_custom; m = build_cnn_custom(); m.summary()` works
- `from src.models.mobilenet_finetune import build_mobilenet_head; m = build_mobilenet_head(); m.summary()` works
- No notebooks touched yet

## Notes

- No training logic inside the model builders — only architecture and compilation
- Type hints required in all public functions
- Google-style docstrings on every public function

## Output

Importable modules. Next phase: Notebook 1 (EDA) that consumes the loader.
