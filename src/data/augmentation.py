from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_augmentation_pipeline(
    for_minority_classes: bool = False,
) -> ImageDataGenerator:
    """Build a Keras ImageDataGenerator for training-time augmentation.

    The standard pipeline mirrors realistic webcam variation: small rotations
    (head tilt), slight shifts, mild zoom, brightness changes, and horizontal
    flip. Vertical flip is intentionally omitted — faces are not vertically
    symmetric.

    Args:
        for_minority_classes: If True, returns a stronger pipeline with higher
            rotation and zoom to help under-represented classes (e.g. Disgust,
            Fear) generalise better without simply duplicating low-quality images.

    Returns:
        A configured ImageDataGenerator (not yet fit to data).
    """
    if for_minority_classes:
        return ImageDataGenerator(
            rotation_range=25,
            width_shift_range=0.15,
            height_shift_range=0.15,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=(0.7, 1.3),
            fill_mode="nearest",
        )
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
        fill_mode="nearest",
    )
