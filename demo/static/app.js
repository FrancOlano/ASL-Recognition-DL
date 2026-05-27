const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const clearBtn = document.getElementById("clearBtn");
const backspaceBtn = document.getElementById("backspaceBtn");
const spaceBtn = document.getElementById("spaceBtn");
const cameraState = document.getElementById("cameraState");
const predictionLetter = document.getElementById("predictionLetter");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceLabel = document.getElementById("confidenceLabel");
const topK = document.getElementById("topK");
const transcript = document.getElementById("transcript");

const appState = {
  stream: null,
  timer: null,
  lastPrediction: null,
  stableCount: 0,
  commitLockUntil: 0,
};

const STABLE_FRAMES = 5;
const CONFIDENCE_THRESHOLD = 0.82;
const COMMIT_COOLDOWN_MS = 900;
const PREDICT_INTERVAL_MS = 240;

function setCameraState(text, isReady = false) {
  cameraState.textContent = text;
  cameraState.style.color = isReady ? "var(--accent)" : "var(--muted)";
}

function renderTopK(items) {
  topK.innerHTML = "";
  items.forEach((item, index) => {
    const li = document.createElement("li");
    li.innerHTML = `<span>${index + 1}. ${item.letter}</span><strong>${Math.round(item.confidence * 100)}%</strong>`;
    topK.appendChild(li);
  });
}

function updatePredictionUI(letter, confidence, top3 = []) {
  predictionLetter.textContent = letter || "-";
  confidenceFill.style.width = `${Math.max(0, Math.min(100, confidence * 100))}%`;
  confidenceLabel.textContent = `${Math.round(confidence * 100)}%`;
  renderTopK(top3);
}

function appendTranscript(text) {
  transcript.value += text;
  transcript.focus();
  transcript.setSelectionRange(transcript.value.length, transcript.value.length);
}

function commitCharacter(character) {
  const now = Date.now();
  if (now < appState.commitLockUntil) {
    return;
  }

  appendTranscript(character);
  appState.commitLockUntil = now + COMMIT_COOLDOWN_MS;
}

function processPrediction(payload) {
  if (!payload.ok) {
    setCameraState(payload.error || "Prediction unavailable", false);
    updatePredictionUI("-", 0, []);
    return;
  }

  const { prediction, confidence, top3 } = payload;
  updatePredictionUI(prediction, confidence, top3);

  if (confidence < CONFIDENCE_THRESHOLD) {
    appState.lastPrediction = null;
    appState.stableCount = 0;
    return;
  }

  if (prediction === appState.lastPrediction) {
    appState.stableCount += 1;
  } else {
    appState.lastPrediction = prediction;
    appState.stableCount = 1;
  }

  setCameraState(`Tracking ${prediction}`, true);

  if (appState.stableCount >= STABLE_FRAMES) {
    commitCharacter(prediction);
    appState.stableCount = 0;
  }
}

async function captureAndPredict() {
  if (!appState.stream || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    return;
  }

  const context = canvas.getContext("2d", { willReadFrequently: false });
  context.save();
  context.translate(canvas.width, 0);
  context.scale(-1, 1);
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  context.restore();

  const image = canvas.toDataURL("image/jpeg", 0.78);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image }),
    });

    const payload = await response.json();
    processPrediction(payload);
  } catch (error) {
    setCameraState("Prediction request failed", false);
  }
}

async function startCamera() {
  try {
    appState.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 1280 },
      },
      audio: false,
    });

    video.srcObject = appState.stream;
    await video.play();
    setCameraState("Camera active", true);

    if (appState.timer) {
      clearInterval(appState.timer);
    }

    appState.timer = setInterval(captureAndPredict, PREDICT_INTERVAL_MS);
  } catch (error) {
    setCameraState("Camera permission denied", false);
  }
}

function stopCamera() {
  if (appState.timer) {
    clearInterval(appState.timer);
    appState.timer = null;
  }

  if (appState.stream) {
    appState.stream.getTracks().forEach((track) => track.stop());
    appState.stream = null;
  }

  video.srcObject = null;
  setCameraState("Camera stopped", false);
}

function clearTranscript() {
  transcript.value = "";
}

function backspaceTranscript() {
  transcript.value = transcript.value.slice(0, -1);
}

function addSpace() {
  if (!transcript.value.endsWith(" ") && transcript.value.length > 0) {
    transcript.value += " ";
  }
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
clearBtn.addEventListener("click", clearTranscript);
backspaceBtn.addEventListener("click", backspaceTranscript);
spaceBtn.addEventListener("click", addSpace);

updatePredictionUI("-", 0, []);
setCameraState(window.ASL_DEMO?.modelReady ? "Ready to start" : "Add a checkpoint to enable inference", window.ASL_DEMO?.modelReady);

if (window.ASL_DEMO?.statusMessage && !window.ASL_DEMO.modelReady) {
  topK.innerHTML = `<li><span>${window.ASL_DEMO.statusMessage}</span><strong>setup</strong></li>`;
}