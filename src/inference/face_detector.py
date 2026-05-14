from typing import List, Optional, Tuple

import cv2
import numpy as np


class FaceDetector:
    """Haar-cascade face detector wrapping OpenCV's frontal-face classifier.

    Designed for the sales-call demo scenario: a single person fills most of
    the frame, so ``detect_largest`` is the primary entry point. Using Haar
    cascades keeps the dependency footprint minimal and avoids the occasional
    Windows-specific issues seen with MediaPipe.
    """

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
    ) -> None:
        """Load the pre-bundled Haar frontal-face cascade.

        Args:
            scale_factor: Image scale reduction at each pyramid level.
            min_neighbors: Minimum neighbour rectangles required to retain
                a candidate detection.
            min_size: Minimum face size in pixels (width, height).
        """
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size = min_size

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Return all detected face bounding boxes.

        Args:
            image: BGR or grayscale image as a uint8 numpy array.

        Returns:
            List of ``(x, y, w, h)`` tuples, one per detected face.
            Empty list if no face is found.
        """
        gray = self._to_gray(image)
        detections = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
        )
        if len(detections) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in detections]

    def detect_largest(
        self, image: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """Return the bounding box of the largest detected face.

        In a sales-call frame the primary subject typically occupies the most
        area, so the largest detection is the most relevant one.

        Args:
            image: BGR or grayscale image as a uint8 numpy array.

        Returns:
            ``(x, y, w, h)`` of the largest face, or ``None`` if no face found.
        """
        boxes = self.detect(image)
        if not boxes:
            return None
        return max(boxes, key=lambda b: b[2] * b[3])

    def extract_roi(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        target_size: Tuple[int, int],
        to_grayscale: bool = True,
    ) -> np.ndarray:
        """Crop the face region and resize it to ``target_size``.

        Args:
            image: Source image (BGR or grayscale, uint8).
            bbox: ``(x, y, w, h)`` bounding box returned by ``detect*``.
            target_size: ``(height, width)`` of the output patch.
            to_grayscale: If ``True``, convert the crop to grayscale before
                resizing. Set to ``False`` when feeding a 3-channel model.

        Returns:
            Cropped, resized face as a uint8 array of shape
            ``(height, width)`` if ``to_grayscale`` else ``(height, width, 3)``.
        """
        x, y, w, h = bbox
        crop = image[y : y + h, x : x + w]
        if to_grayscale:
            crop = self._to_gray(crop)
        th, tw = target_size
        resized = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_LINEAR)
        return resized

    # ------------------------------------------------------------------
    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        if image.shape[2] == 1:
            return image[:, :, 0]
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
