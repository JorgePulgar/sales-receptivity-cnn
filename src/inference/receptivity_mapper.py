from collections import deque
from typing import List

from src import config


def map_emotion_to_signal(emotion: str) -> str:
    """Return the human-readable commercial signal for an emotion label.

    Args:
        emotion: One of the keys in ``config.EMOTION_TO_SIGNAL``.

    Returns:
        A short descriptive string suitable for overlay display.
    """
    return config.EMOTION_TO_SIGNAL[emotion]


def map_emotion_to_score(emotion: str) -> float:
    """Return the receptivity score (0–10) for an emotion label.

    Args:
        emotion: One of the keys in ``config.EMOTION_TO_SCORE``.

    Returns:
        Float score where high values indicate buyer engagement.
    """
    return config.EMOTION_TO_SCORE[emotion]


class ReceptivityIndex:
    """Maintains a rolling weighted average of receptivity scores.

    Each call to ``update`` appends an emotion observation to a fixed-length
    buffer and returns the current index. When no face is detected the caller
    should skip ``update``; the index retains its last value until the next
    valid observation arrives.

    Args:
        window_size: Number of most recent observations to consider.
        weight_by_confidence: If ``True``, each score is weighted by the
            classifier confidence before averaging, giving higher-certainty
            predictions more influence on the index.
    """

    def __init__(
        self,
        window_size: int = 10,
        weight_by_confidence: bool = True,
    ) -> None:
        self._window_size = window_size
        self._weight_by_confidence = weight_by_confidence
        self._scores: deque = deque(maxlen=window_size)
        self._confidences: deque = deque(maxlen=window_size)
        self._history: List[float] = []

    def update(self, emotion: str, confidence: float) -> float:
        """Record one observation and return the updated index.

        Args:
            emotion: Predicted emotion label (key in ``config.EMOTION_TO_SCORE``).
            confidence: Classifier softmax probability for the predicted class.

        Returns:
            Current receptivity index in [0, 10].
        """
        self._scores.append(map_emotion_to_score(emotion))
        self._confidences.append(confidence)
        index = self.get_current_index()
        self._history.append(index)
        return index

    def get_current_index(self) -> float:
        """Return the current receptivity index without recording a new observation.

        Returns:
            Weighted (or unweighted) moving average in [0, 10].
            Returns 5.0 (neutral baseline) if no observations have been recorded.
        """
        if not self._scores:
            return 5.0
        scores = list(self._scores)
        if self._weight_by_confidence:
            weights = list(self._confidences)
            total_weight = sum(weights)
            if total_weight == 0:
                return float(sum(scores) / len(scores))
            return float(sum(s * w for s, w in zip(scores, weights)) / total_weight)
        return float(sum(scores) / len(scores))

    def reset(self) -> None:
        """Clear the observation buffer and history."""
        self._scores.clear()
        self._confidences.clear()
        self._history.clear()

    def get_history(self) -> List[float]:
        """Return the full list of index values recorded since last reset.

        Useful for plotting the receptivity trend over a session.
        """
        return list(self._history)
