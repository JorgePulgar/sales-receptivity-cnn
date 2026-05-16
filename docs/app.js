'use strict';

// ── Constants ──────────────────────────────────────────────────────────────────

const LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'];

// Mirrors config.EMOTION_TO_SCORE
const EMOTION_SCORES = {
  happy: 9.0, surprise: 7.0, neutral: 5.0,
  sad: 3.0, fear: 2.0, angry: 1.0, disgust: 1.0,
};

// Mirrors config.EMOTION_TO_SIGNAL
const EMOTION_SIGNALS = {
  happy:    'Positive — reinforce proposal',
  surprise: 'Interested — elaborate on point',
  neutral:  'Attentive — continue normally',
  sad:      'Disengaged — check in with prospect',
  fear:     'Uncomfortable — slow down, clarify',
  angry:    'Resistant — pause and address concern',
  disgust:  'Resistant — pause and address concern',
};

const EMOTION_COLORS = {
  happy: '#16a34a', surprise: '#0891b2', neutral: '#4f46e5',
  sad: '#7c3aed', fear: '#b45309', angry: '#dc2626', disgust: '#991b1b',
};

// Model input specs — must match EmotionClassifier constructor args in Python.
const MODEL_CONFIG = {
  mobilenet_ft: { h: 96, w: 96, grayscale: false },
  cnn_custom:   { h: 48, w: 48, grayscale: true  },
};

const TARGET_FPS = 8;
const FRAME_MS   = 1000 / TARGET_FPS;
const CHART_SECS = 30;

// ── ReceptivityIndex ──────────────────────────────────────────────────────────
// JS port of src/inference/receptivity_mapper.py :: ReceptivityIndex.
// Identical rolling weighted-average logic and emotion→score mapping.

class ReceptivityIndex {
  constructor(windowSize = 10, weightByConfidence = true) {
    this._maxWin  = windowSize;
    this._useConf = weightByConfidence;
    this._scores  = [];
    this._confs   = [];
    this._history = [];
  }

  update(emotion, confidence) {
    if (this._scores.length >= this._maxWin) {
      this._scores.shift();
      this._confs.shift();
    }
    this._scores.push(EMOTION_SCORES[emotion] ?? 5.0);
    this._confs.push(confidence);
    const idx = this.currentIndex;
    this._history.push(idx);
    return idx;
  }

  get currentIndex() {
    if (!this._scores.length) return 5.0;
    if (this._useConf) {
      const totalW = this._confs.reduce((s, w) => s + w, 0);
      if (totalW === 0)
        return this._scores.reduce((s, v) => s + v, 0) / this._scores.length;
      return this._scores.reduce((s, v, i) => s + v * this._confs[i], 0) / totalW;
    }
    return this._scores.reduce((s, v) => s + v, 0) / this._scores.length;
  }

  setWindowSize(n) {
    this._maxWin = n;
    while (this._scores.length > n) { this._scores.shift(); this._confs.shift(); }
  }

  reset() { this._scores = []; this._confs = []; this._history = []; }
  get history() { return this._history; }
}

// ── Face detection ────────────────────────────────────────────────────────────
// Uses BlazeFace loaded from CDN.
//
// Detector mismatch note: the Streamlit demo uses OpenCV Haar Cascades; this
// demo uses BlazeFace.  Bounding boxes differ between the two detectors, so
// cropped ROIs differ and predictions will not be bit-identical across demos.
// This is expected — it is not a model bug.

let faceDetector = null;

async function loadFaceDetector() {
  faceDetector = await blazeface.load();
}

// Returns { x, y, w, h } for the largest face in the frame, or null.
async function detectFace(video) {
  if (!faceDetector || video.readyState < 2) return null;
  const preds = await faceDetector.estimateFaces(video, /* returnTensors */ false);
  if (!preds || !preds.length) return null;

  let best = null, bestArea = -1;
  for (const p of preds) {
    const [x1, y1] = p.topLeft;
    const [x2, y2] = p.bottomRight;
    const area = (x2 - x1) * (y2 - y1);
    if (area > bestArea) { bestArea = area; best = p; }
  }
  if (!best) return null;

  const [x1, y1] = best.topLeft;
  const [x2, y2] = best.bottomRight;
  return { x: Math.max(0, x1), y: Math.max(0, y1), w: x2 - x1, h: y2 - y1 };
}
