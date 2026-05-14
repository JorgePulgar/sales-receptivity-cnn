# demo/app.py
# This module imports src.inference directly instead of routing through the
# FastAPI endpoint. Running two separate processes (uvicorn + streamlit) during
# a class presentation adds failure points and synchronisation overhead without
# any benefit: both processes share the same filesystem and the same GPU, so
# calling the endpoint would just be a local HTTP round-trip. Direct import
# keeps the demo self-contained and eliminates network errors during live demos.

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.inference.face_detector import FaceDetector
from src.inference.emotion_classifier import EmotionClassifier
from src.inference.receptivity_mapper import ReceptivityIndex, map_emotion_to_score


def _resolve_model():
    """Return (path, input_size, use_rgb) for the best available model, or Nones."""
    candidates = [
        (config.MODELS_DIR / "mobilenet_ft.keras", config.IMG_SIZE_MOBILENET, True),
        (config.MODELS_DIR / "mobilenet_ft.h5",    config.IMG_SIZE_MOBILENET, True),
        (config.MODELS_DIR / "cnn_custom.keras",   config.IMG_SIZE_CUSTOM, False),
        (config.MODELS_DIR / "cnn_custom.h5",      config.IMG_SIZE_CUSTOM, False),
    ]
    for p, size, rgb in candidates:
        if p.exists():
            return p, size, rgb
    return None, None, None

st.sidebar.title("Sales Receptivity CNN")
st.sidebar.markdown(
    "Emotion-based receptivity analyser for sales presentations. "
    "The CNN predicts the prospect's emotional state frame by frame and "
    "aggregates it into a rolling receptivity index."
)

mode = st.sidebar.selectbox("Mode", ["Recorded video", "Webcam"])
window_size = st.sidebar.slider(
    "Receptivity window size", min_value=5, max_value=30, value=10
)
weight_by_confidence = st.sidebar.checkbox("Weight by confidence", value=True)

_model_path, _input_size, _use_rgb = _resolve_model()
if _model_path is not None:
    st.sidebar.info(
        f"**Model loaded:** `{_model_path.name}`  \n"
        f"Input size: {_input_size[1]}×{_input_size[0]} px"
    )
else:
    st.sidebar.warning(
        "No trained model found in `models/`.  \n"
        "Train a model in Notebook 3 first."
    )

# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — Recorded video
# ══════════════════════════════════════════════════════════════════════════════

if mode == "Recorded video":
    st.title("Recorded Video Analysis")
    uploaded = st.file_uploader(
        "Upload a sales recording", type=["mp4", "avi", "mov"]
    )

    if uploaded is not None:
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)

        cap = cv2.VideoCapture(str(tmp_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        sample_step = max(1, int(fps))
        sample_indices = list(range(0, total_frames, sample_step))
        n_samples = len(sample_indices)

        if _model_path is None:
            cap.release()
            tmp_path.unlink(missing_ok=True)
            st.error("No model loaded — train one in Notebook 3.")
            st.stop()

        detector = FaceDetector()
        classifier = EmotionClassifier(_model_path, _input_size, _use_rgb)
        ri = ReceptivityIndex(window_size=window_size, weight_by_confidence=weight_by_confidence)

        progress_bar = st.progress(0, text="Analysing frames…")
        records: list[dict] = []
        face_misses = 0
        key_frames: dict[int, tuple] = {}

        for step, frame_idx in enumerate(sample_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                face_misses += 1
                progress_bar.progress((step + 1) / n_samples)
                continue
            bbox = detector.detect_largest(frame)
            if bbox is None:
                face_misses += 1
                progress_bar.progress((step + 1) / n_samples)
                continue
            roi = detector.extract_roi(
                frame, bbox, _input_size, to_grayscale=not _use_rgb
            )
            emotion, confidence, _ = classifier.predict(roi)
            idx_val = ri.update(emotion, confidence)
            rec_idx = len(records)
            records.append(
                {
                    "timestamp": round(frame_idx / fps, 2),
                    "emotion": emotion,
                    "confidence": round(confidence, 3),
                    "score": map_emotion_to_score(emotion),
                    "index_value": round(idx_val, 3),
                }
            )
            key_frames[rec_idx] = (frame.copy(), bbox)
            progress_bar.progress((step + 1) / n_samples)

        cap.release()
        tmp_path.unlink(missing_ok=True)

        if not records:
            st.warning("No faces detected in the video.")
            st.stop()

        df = pd.DataFrame(records)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Receptivity Timeline", "Session Summary", "Frame-by-Frame", "Key Moments"]
        )

        _EMOTION_COLORS = {
            "happy":    "rgba(34,197,94,0.18)",
            "surprise": "rgba(59,130,246,0.18)",
            "neutral":  "rgba(156,163,175,0.12)",
            "sad":      "rgba(99,102,241,0.18)",
            "fear":     "rgba(168,85,247,0.18)",
            "angry":    "rgba(239,68,68,0.18)",
            "disgust":  "rgba(180,83,9,0.18)",
        }

        def _show_key_frames(subset: pd.DataFrame, border_bgr: tuple) -> None:
            cols = st.columns(min(3, len(subset)))
            for col, (rec_i, row) in zip(cols, subset.iterrows()):
                if rec_i in key_frames:
                    frm, bbox = key_frames[rec_i]
                    frm = frm.copy()
                    x, y, w, h = bbox
                    cv2.rectangle(frm, (x, y), (x + w, y + h), border_bgr, 2)
                    col.image(
                        cv2.cvtColor(frm, cv2.COLOR_BGR2RGB),
                        caption=(
                            f"{row['emotion'].capitalize()} "
                            f"({row['confidence']:.0%}) | "
                            f"{row['timestamp']}s | "
                            f"idx {row['index_value']:.2f}"
                        ),
                    )

        with tab4:
            st.subheader("3 Highest-Receptivity Frames")
            _show_key_frames(df.nlargest(3, "index_value"), (0, 200, 80))
            st.subheader("3 Lowest-Receptivity Frames")
            _show_key_frames(df.nsmallest(3, "index_value"), (50, 50, 220))

        with tab3:
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download CSV",
                data=df.to_csv(index=False),
                file_name="receptivity_analysis.csv",
                mime="text/csv",
            )

        with tab2:
            dominant = df["emotion"].value_counts().idxmax()
            mean_rec = df["index_value"].mean()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Dominant emotion", dominant.capitalize())
            c2.metric("Mean receptivity", f"{mean_rec:.2f} / 10")
            c3.metric("Frames analysed", len(df))
            c4.metric("Frames without face", face_misses)

            emotion_counts = df["emotion"].value_counts()
            pie = go.Figure(
                go.Pie(
                    labels=[e.capitalize() for e in emotion_counts.index],
                    values=emotion_counts.values.tolist(),
                    hole=0.32,
                )
            )
            pie.update_layout(title="Time Spent per Emotion", height=370)
            st.plotly_chart(pie, use_container_width=True)

        with tab1:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["index_value"],
                    mode="lines+markers",
                    name="Receptivity",
                    line=dict(color="#2563eb", width=2),
                )
            )

            prev_em, seg_start = None, None
            for _, row in df.iterrows():
                if row["emotion"] != prev_em:
                    if prev_em is not None:
                        fig.add_vrect(
                            x0=seg_start,
                            x1=row["timestamp"],
                            fillcolor=_EMOTION_COLORS.get(prev_em, "rgba(0,0,0,0.05)"),
                            line_width=0,
                            annotation_text=prev_em,
                            annotation_position="top left",
                        )
                    seg_start = row["timestamp"]
                    prev_em = row["emotion"]
            if prev_em is not None:
                fig.add_vrect(
                    x0=seg_start,
                    x1=df["timestamp"].iloc[-1],
                    fillcolor=_EMOTION_COLORS.get(prev_em, "rgba(0,0,0,0.05)"),
                    line_width=0,
                    annotation_text=prev_em,
                    annotation_position="top left",
                )

            for i in df["index_value"].nlargest(3).index:
                fig.add_annotation(
                    x=df.loc[i, "timestamp"],
                    y=df.loc[i, "index_value"],
                    text="▲ peak",
                    showarrow=True,
                    arrowhead=2,
                    yanchor="bottom",
                    font=dict(color="#16a34a"),
                )
            for i in df["index_value"].nsmallest(3).index:
                fig.add_annotation(
                    x=df.loc[i, "timestamp"],
                    y=df.loc[i, "index_value"],
                    text="▼ valley",
                    showarrow=True,
                    arrowhead=2,
                    yanchor="top",
                    font=dict(color="#dc2626"),
                )

            fig.update_layout(
                title="Receptivity Index Over Time",
                xaxis_title="Time (s)",
                yaxis_title="Receptivity Index (0–10)",
                yaxis=dict(range=[0, 10]),
                height=460,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                "The receptivity index is a rolling weighted average of per-emotion scores "
                "(0–10 scale). Values above 6 indicate positive engagement; below 4 signal "
                "discomfort or disengagement. Shaded bands mark dominant-emotion segments; "
                "annotations mark the three highest and lowest peaks."
            )

# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — Webcam
# ══════════════════════════════════════════════════════════════════════════════

elif mode == "Webcam":
    st.title("Live Webcam Analysis")

    photo = st.camera_input("Take a photo")

    if photo is not None:
        if _model_path is None:
            st.error("No model loaded — train one in Notebook 3.")
        else:
            raw = np.frombuffer(photo.getvalue(), np.uint8)
            frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)

            cam_detector = FaceDetector()
            cam_classifier = EmotionClassifier(_model_path, _input_size, _use_rgb)
            cam_ri = ReceptivityIndex(
                window_size=window_size, weight_by_confidence=weight_by_confidence
            )

            bbox = cam_detector.detect_largest(frame)
            if bbox is None:
                st.warning("No face detected — try better lighting or move closer.")
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                x, y, w, h = bbox
                display = frame.copy()
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 220, 80), 2)
                st.image(
                    cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                    caption="Detected face",
                    use_column_width=True,
                )
                roi = cam_detector.extract_roi(
                    frame, bbox, _input_size, to_grayscale=not _use_rgb
                )
                emotion, confidence, _ = cam_classifier.predict(roi)
                idx_val = cam_ri.update(emotion, confidence)

                st.markdown(
                    f"**Predicted emotion:** {emotion.capitalize()}  \n"
                    f"**Confidence:** {confidence:.1%}"
                )

                st.metric("Current Receptivity Index", f"{idx_val:.2f} / 10")

                if "emotion_history" not in st.session_state:
                    st.session_state.emotion_history = []
                st.session_state.emotion_history.append(emotion)
                hist = {
                    e: st.session_state.emotion_history.count(e)
                    for e in config.EMOTION_LABELS
                }
                st.bar_chart(hist)
