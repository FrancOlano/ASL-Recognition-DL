const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const clearBtn = document.getElementById("clearBtn");
const backspaceBtn = document.getElementById("backspaceBtn");
const spaceBtn = document.getElementById("spaceBtn");
const modelSelect = document.getElementById("modelSelect");
const loadModelBtn = document.getElementById("loadModelBtn");
const cameraState = document.getElementById("cameraState");
const predictionLetter = document.getElementById("predictionLetter");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceLabel = document.getElementById("confidenceLabel");
const topK = document.getElementById("topK");
const transcript = document.getElementById("transcript");
const currentModelLabel = document.getElementById("currentModelLabel");
const currentModelArchitecture = document.getElementById("currentModelArchitecture");
const currentModelCheckpoint = document.getElementById("currentModelCheckpoint");
const statusMessage = document.getElementById("statusMessage");
const modelSwitchNote = document.getElementById("modelSwitchNote");
const leftHandBtn = document.getElementById("leftHandBtn");
const rightHandBtn = document.getElementById("rightHandBtn");

// Offscreen canvas for downscaling images (drastically reduces memory/payloads)
const processCanvas = document.createElement("canvas");
processCanvas.width = 256; 
processCanvas.height = 256;
const processCtx = processCanvas.getContext("2d", { willReadFrequently: false });

const appState = {
  stream: null,
  timer: null,
  lastPrediction: null,
  stableCount: 0,
  commitLockUntil: 0,
  modelKey: window.ASL_DEMO?.currentModelKey || window.ASL_DEMO?.defaultModelKey || null,
  switchingModel: false,
  isPredicting: false, 
  hand: "switch", // FIXED: Default matches the Left hand button state
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

function setModelSwitchState(message, isLoading = false) {
  if (modelSwitchNote) modelSwitchNote.textContent = message;
  if (loadModelBtn) loadModelBtn.disabled = isLoading;
  if (modelSelect) modelSelect.disabled = isLoading;
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
  if (now < appState.commitLockUntil) return;
  appendTranscript(character);
  appState.commitLockUntil = now + COMMIT_COOLDOWN_MS;
}

function commitAction(action) {
  const now = Date.now();
  if (now < appState.commitLockUntil) return;
  action();
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

  if (confidence < 0.7) {
    appState.lastPrediction = null;
    appState.stableSince = null;
    return;
  }

  if (prediction !== appState.lastPrediction) {
    appState.lastPrediction = prediction;
    appState.stableSince = Date.now();
  }

  setCameraState(`Tracking ${prediction}`, true);

  const now = Date.now();
  const timeStable = now - appState.stableSince;

  let requiredFirstWriteDelay = 1200;
  if (confidence >= 0.95) requiredFirstWriteDelay = 500;
  else if (confidence >= 0.85) requiredFirstWriteDelay = 800;

  if (timeStable >= requiredFirstWriteDelay) {
    const normalized = String(prediction || "").toLowerCase();
    
    if (normalized === "nothing") {
      appState.lastPrediction = null;
      appState.stableSince = null;
      return;
    }
    
    if (normalized === "space") return commitAction(addSpace);
    if (normalized === "del" || normalized === "backspace") return commitAction(backspaceTranscript);

    commitCharacter(prediction);
  }
}

async function captureAndPredict() {
  if (appState.switchingModel || appState.isPredicting) return;
  if (!appState.stream || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;

  appState.isPredicting = true; 

  try {
    // 1. Draw to visual UI canvas as normal (mirrored for the user)
    const context = canvas.getContext("2d", { willReadFrequently: false });
    context.save();
    context.translate(canvas.width, 0);
    context.scale(-1, 1);
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    context.restore();

    // 2. Downscale into smaller processCanvas to avoid Base64 memory bloat
    const minDim = Math.min(video.videoWidth, video.videoHeight);
    const startX = (video.videoWidth - minDim) / 2;
    const startY = (video.videoHeight - minDim) / 2;

    processCtx.save();
    
    // Flip horizontally if the Left hand is selected.
    // This makes a left hand look like a right hand to the model.
    if (appState.hand === "switch") {
      processCtx.translate(processCanvas.width, 0);
      processCtx.scale(-1, 1);
    }
    
    processCtx.drawImage(
      video,
      startX, startY, minDim, minDim,         
      0, 0, processCanvas.width, processCanvas.height 
    );
    processCtx.restore();

    const image = processCanvas.toDataURL("image/jpeg", 0.7);

    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image }),
    });

    const payload = await response.json();
    processPrediction(payload);
  } catch (error) {
    setCameraState("Prediction request failed", false);
  } finally {
    appState.isPredicting = false;
  }
}

async function loadSelectedModel() {
  const selectedModel = modelSelect?.value;
  if (!selectedModel) return setModelSwitchState("No model is available to load.");
  if (selectedModel === appState.modelKey) return setModelSwitchState("Selected model is already active.");

  appState.switchingModel = true;
  setModelSwitchState(`Loading ${selectedModel}...`, true);

  try {
    const response = await fetch("/api/model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: selectedModel }),
    });

    const payload = await response.json();
    if (!payload.ok) throw new Error(payload.model?.message || payload.error || "Model switch failed");

    appState.modelKey = payload.model.key;
    if (currentModelLabel) currentModelLabel.textContent = payload.model.label || payload.model.key;
    if (currentModelArchitecture) currentModelArchitecture.textContent = payload.model.architecture || "pending";
    if (currentModelCheckpoint) currentModelCheckpoint.textContent = payload.model.checkpoint || "none found";
    if (statusMessage) statusMessage.textContent = `Loaded ${payload.model.label}`;

    setCameraState(`Using ${payload.model.label}`, true);
    setModelSwitchState(`Active model: ${payload.model.label}`);
  } catch (error) {
    if (modelSelect) modelSelect.value = appState.modelKey || window.ASL_DEMO?.defaultModelKey || "";
    setModelSwitchState(error.message || "Failed to load model");
    setCameraState("Model switch failed", false);
  } finally {
    appState.switchingModel = false;
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

    if (appState.timer) clearInterval(appState.timer);
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

function clearTranscript() { transcript.value = ""; }
function backspaceTranscript() { transcript.value = transcript.value.slice(0, -1); }
function addSpace() {
  if (!transcript.value.endsWith(" ") && transcript.value.length > 0) {
    transcript.value += " ";
  }
}

// Event Listeners
startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
clearBtn.addEventListener("click", clearTranscript);
backspaceBtn.addEventListener("click", backspaceTranscript);
spaceBtn.addEventListener("click", addSpace);

leftHandBtn?.addEventListener("click", () => {
  appState.hand = "switch";
  leftHandBtn.classList.add("active");
  rightHandBtn.classList.remove("active");
});

rightHandBtn?.addEventListener("click", () => {
  appState.hand = "no-switch";
  rightHandBtn.classList.add("active");
  leftHandBtn.classList.remove("active");
});

loadModelBtn?.addEventListener("click", loadSelectedModel);
modelSelect?.addEventListener("change", () => {
  const selectedLabel = modelSelect.options[modelSelect.selectedIndex]?.textContent?.trim() || modelSelect.value;
  setModelSwitchState(`Selected ${selectedLabel}. Click Load selected to apply.`);
});

// Initial Setup
updatePredictionUI("-", 0, []);
setCameraState(window.ASL_DEMO?.modelReady ? "Ready to start" : "Add a checkpoint to enable inference", window.ASL_DEMO?.modelReady);

if (currentModelLabel && window.ASL_DEMO?.currentModelLabel) {
  currentModelLabel.textContent = window.ASL_DEMO.currentModelLabel;
}

if (modelSelect && appState.modelKey) {
  modelSelect.value = appState.modelKey;
}

setModelSwitchState(
  window.ASL_DEMO?.currentModelLabel
    ? `Active model: ${window.ASL_DEMO.currentModelLabel}`
    : "Switching models updates inference on the next capture."
);

if (window.ASL_DEMO?.statusMessage && !window.ASL_DEMO.modelReady) {
  topK.innerHTML = `<li><span>${window.ASL_DEMO.statusMessage}</span><strong>setup</strong></li>`;
}