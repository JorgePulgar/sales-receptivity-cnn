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

// ── Preprocessing ─────────────────────────────────────────────────────────────
// Must match EmotionClassifier._preprocess in Python exactly:
//   resize → cv2.INTER_LINEAR = bilinear (tf.image.resizeBilinear default)
//   normalise → / 255.0
//   grayscale → BT.601 luma (matches cv2.cvtColor RGB→GRAY on RGB input)
// Call inside tf.tidy() — caller owns tensor lifetime.

// Reused offscreen canvas avoids repeated DOM allocation.
const offCanvas = document.createElement('canvas');
const offCtx    = offCanvas.getContext('2d', { willReadFrequently: true });

function buildInputTensor(video, bbox, modelCfg) {
  const { h: th, w: tw, grayscale } = modelCfg;

  offCanvas.width  = tw;
  offCanvas.height = th;
  offCtx.drawImage(video, bbox.x, bbox.y, bbox.w, bbox.h, 0, 0, tw, th);

  let t = tf.browser.fromPixels(offCanvas, 3);       // (H, W, 3) uint8
  t = tf.cast(t, 'float32');
  t = tf.div(t, 255.0);                              // [0, 1]

  if (grayscale) {
    // 0.299 R + 0.587 G + 0.114 B — matches OpenCV cvtColor(RGB→GRAY)
    const luma = tf.tensor1d([0.299, 0.587, 0.114]);
    t = tf.sum(tf.mul(t, luma), /* axis */ -1, /* keepDims */ true); // (H, W, 1)
  }

  return tf.expandDims(t, 0);                        // (1, H, W, C)
}

// ── State ────────────────────────────────────────────────────────────────────

let emotionModel   = null;
let receptivity    = new ReceptivityIndex(10);
let rafId          = null;
let lastFrameMs    = 0;
let currentModel   = 'mobilenet_ft';
let isLoadingModel = false;
let isProcessing   = false;

const emotionCounts = Object.fromEntries(LABELS.map(l => [l, 0]));
const riHistory     = [];  // [{ t: seconds, v: index }]

// ── Model loading ─────────────────────────────────────────────────────────────

async function loadEmotionModel(name) {
  if (isLoadingModel) return;
  isLoadingModel = true;
  emotionModel   = null;
  setStatus(`Loading ${name}…`);

  try {
    emotionModel = await tf.loadLayersModel(`models/${name}/model.json`);
    setStatus('');
  } catch (e) {
    setStatus(`Failed to load ${name}: ${e.message}`, true);
  } finally {
    isLoadingModel = false;
  }

  LABELS.forEach(l => (emotionCounts[l] = 0));
  receptivity.reset();
  riHistory.length = 0;
}

// ── Per-frame inference pipeline ─────────────────────────────────────────────

async function processFrame(now) {
  rafId = requestAnimationFrame(processFrame);

  if (now - lastFrameMs < FRAME_MS || isProcessing) return;
  lastFrameMs  = now;
  isProcessing = true;

  try {
    const vw = videoEl.videoWidth  || 640;
    const vh = videoEl.videoHeight || 480;
    if (canvasEl.width  !== vw) canvasEl.width  = vw;
    if (canvasEl.height !== vh) canvasEl.height = vh;
    ctx.drawImage(videoEl, 0, 0);

    if (!emotionModel || isLoadingModel) return;

    const bbox = await detectFace(videoEl);

    if (!bbox) {
      ctx.fillStyle = '#f59e0b';
      ctx.font      = '13px monospace';
      ctx.fillText('No face detected', 10, 22);
      return;
    }

    ctx.strokeStyle = '#6366f1';
    ctx.lineWidth   = 2;
    ctx.strokeRect(bbox.x, bbox.y, bbox.w, bbox.h);

    // Preprocess + predict — tf.tidy disposes all intermediate tensors.
    // dataSync() extracts the softmax values before tidy runs cleanup.
    const config   = MODEL_CONFIG[currentModel];
    const probsRaw = tf.tidy(() => {
      const t   = buildInputTensor(videoEl, bbox, config);
      const out = emotionModel.predict(t);
      return out.dataSync();
    });

    const probs   = Array.from(probsRaw);
    const maxIdx  = probs.indexOf(Math.max(...probs));
    const emotion = LABELS[maxIdx];
    const conf    = probs[maxIdx];

    const ri = receptivity.update(emotion, conf);
    emotionCounts[emotion]++;

    const tSec = now / 1000;
    riHistory.push({ t: tSec, v: ri });
    const cutoff = tSec - CHART_SECS;
    while (riHistory.length > 1 && riHistory[0].t < cutoff) riHistory.shift();

    // Emotion label overlay
    const badge = `${emotion} ${(conf * 100).toFixed(0)}%`;
    ctx.font      = 'bold 13px monospace';
    const labelW  = ctx.measureText(badge).width + 10;
    const labelY  = Math.max(0, bbox.y - 22);
    ctx.fillStyle = EMOTION_COLORS[emotion] ?? '#6366f1';
    ctx.fillRect(bbox.x, labelY, labelW, 22);
    ctx.fillStyle = '#fff';
    ctx.fillText(badge, bbox.x + 5, labelY + 15);

    updateStats(emotion, conf, ri);
  } finally {
    isProcessing = false;
  }
}

// ── UI updates ────────────────────────────────────────────────────────────────

function updateStats(emotion, conf, ri) {
  emotionBadge.textContent  = emotion;
  emotionBadge.className    = `emotion-badge em-${emotion}`;
  emotionConfEl.textContent = `${(conf * 100).toFixed(1)}% confidence`;
  emotionSigEl.textContent  = EMOTION_SIGNALS[emotion] ?? '';

  riValueEl.textContent = ri.toFixed(1);
  riBarEl.style.width   = `${(ri / 10) * 100}%`;

  drawRiChart();
  drawEmChart();
}

function drawRiChart() {
  const el   = riChartEl;
  const ctx2 = el.getContext('2d');
  el.width   = el.offsetWidth  || 240;
  el.height  = el.offsetHeight || 80;
  const W = el.width, H = el.height;
  ctx2.clearRect(0, 0, W, H);

  if (riHistory.length < 2) return;

  const t0 = riHistory[0].t;
  const dt = (riHistory[riHistory.length - 1].t - t0) || 1;

  // guide lines at 3, 5, 7
  ctx2.strokeStyle = '#2a2d3a';
  ctx2.lineWidth   = 1;
  [3, 5, 7].forEach(v => {
    const y = H - (v / 10) * H;
    ctx2.beginPath(); ctx2.moveTo(0, y); ctx2.lineTo(W, y); ctx2.stroke();
  });

  ctx2.beginPath();
  ctx2.strokeStyle = '#6366f1';
  ctx2.lineWidth   = 2;
  riHistory.forEach(({ t, v }, i) => {
    const x = ((t - t0) / dt) * W;
    const y = H - (v / 10) * H;
    i === 0 ? ctx2.moveTo(x, y) : ctx2.lineTo(x, y);
  });
  ctx2.stroke();
}

function drawEmChart() {
  const el   = emChartEl;
  const ctx2 = el.getContext('2d');
  el.width   = el.offsetWidth  || 240;
  el.height  = el.offsetHeight || 130;
  const W = el.width, H = el.height;
  ctx2.clearRect(0, 0, W, H);

  const total  = LABELS.reduce((s, l) => s + emotionCounts[l], 0) || 1;
  const rowH   = Math.floor(H / LABELS.length);
  const barH   = Math.max(4, rowH - 3);
  const maxBarW = W - 56;

  LABELS.forEach((lbl, i) => {
    const frac = emotionCounts[lbl] / total;
    const y    = i * rowH + 1;
    ctx2.fillStyle = EMOTION_COLORS[lbl];
    ctx2.fillRect(0, y, frac * maxBarW, barH);
    ctx2.fillStyle = '#888';
    ctx2.font      = '10px monospace';
    ctx2.fillText(
      `${lbl.slice(0, 4)} ${(frac * 100).toFixed(0)}%`,
      Math.max(frac * maxBarW + 3, 2),
      y + barH - 2,
    );
  });
}
