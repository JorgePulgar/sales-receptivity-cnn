from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    emotion: str
    confidence: float
    probabilities: Dict[str, float]
    receptivity_signal: str
    receptivity_score: float
    face_detected: bool
    bbox: Optional[Tuple[int, int, int, int]] = None
    inference_time_ms: float


class FrameAnalysis(PredictionResponse):
    frame_index: int


class SessionSummary(BaseModel):
    dominant_emotion: str
    mean_receptivity: float
    time_in_each_state: Dict[str, float]
    peak_frame: int
    valley_frame: int


class SessionResponse(BaseModel):
    frames_analysis: List[FrameAnalysis]
    receptivity_index_over_time: List[float]
    session_summary: SessionSummary
