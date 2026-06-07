// Client-side controller for ASL fingerspelling recognition using ONNX Runtime Web.

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const clearBtn = document.getElementById("clearBtn");
const backspaceBtn = document.getElementById("backspaceBtn");
const spaceBtn = document.getElementById("spaceBtn");
const leftHandBtn = document.getElementById("leftHandBtn");
const rightHandBtn = document.getElementById("rightHandBtn");

const statusTitle = document.getElementById("statusTitle");
const statusMessage = document.getElementById("statusMessage");
const statusDot = document.getElementById("statusDot");
const progressBarContainer = document.getElementById("progressBarContainer");
const progressBarFill = document.getElementById("progressBarFill");
const backendVal = document.getElementById("backendVal");
const cameraBadge = document.getElementById("cameraBadge");
const trackingState = document.getElementById("trackingState");
const predictionLetter = document.getElementById("predictionLetter");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceLabel = document.getElementById("confidenceLabel");
const topK = document.getElementById("topK");
const transcript = document.getElementById("transcript");
const cameraPanel = document.querySelector(".camera-panel");

// Offscreen canvas for cropping and scaling input to model dimensions (200x200)
const preCanvas = document.createElement("canvas");
preCanvas.width = 200;
preCanvas.height = 200;
const preCtx = preCanvas.getContext("2d", { willReadFrequently: true });

// App State
const state = {
  session: null,
  classes: [],
  stream: null,
  timer: null,
  isPredicting: false,
  hand: "left", // Default matches the Left hand button active state
  lastPrediction: null,
  stableSince: null,
  commitLockUntil: 0,
};

const STABLE_COOLDOWN_MS = 900;
const PREDICT_INTERVAL_MS = 200; // Fast 5 FPS predictions inside browser

// Update Status Utility
function setStatus(title, message, dotClass = "") {
  statusTitle.textContent = title;
  statusMessage.textContent = message;
  statusDot.className = "status-dot " + dotClass;
}

// Logit softmax helper (numerically stable)
function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sumExps);
}

// Download ONNX model with progress tracking
async function loadModel() {
  progressBarContainer.style.display = "block";
  setStatus("Downloading Model...", "Fetching model.onnx from folder...", "pulsing");

  try {
    // Fetch classes mapping first
    const classesResponse = await fetch("classes.json");
    if (!classesResponse.ok) throw new Error("Could not load classes.json");
    state.classes = await classesResponse.json();

    // Download model with progress metrics
    const modelResponse = await fetch("model.onnx");
    if (!modelResponse.ok) throw new Error("Failed to load model.onnx");

    const contentLength = modelResponse.headers.get("content-length");
    const totalBytes = contentLength ? parseInt(contentLength, 10) : 0;
    
    if (totalBytes === 0) {
      progressBarContainer.style.display = "none";
      setStatus("Loading Engine...", "Initializing model graph...", "pulsing");
    }

    const reader = modelResponse.body.getReader();
    let loadedBytes = 0;
    const chunks = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      chunks.push(value);
      loadedBytes += value.length;

      if (totalBytes > 0) {
        const percent = Math.round((loadedBytes / totalBytes) * 100);
        progressBarFill.style.width = `${percent}%`;
        setStatus("Downloading Model...", `Loading: ${percent}% (${(loadedBytes / (1024 * 1024)).toFixed(1)} / ${(totalBytes / (1024 * 1024)).toFixed(1)} MB)...`, "pulsing");
      }
    }

    // Combine chunks
    const modelData = new Uint8Array(loadedBytes);
    let position = 0;
    for (const chunk of chunks) {
      modelData.set(chunk, position);
      position += chunk.length;
    }

    progressBarContainer.style.display = "none";
    setStatus("Compiling Model...", "Configuring execution providers...", "pulsing");

    // Initialize ONNX Runtime Session (Request WebGL fallback to WASM)
    state.session = await ort.InferenceSession.create(modelData.buffer, {
      executionProviders: ["webgl", "wasm"],
    });

    let activeProvider = "CPU (WASM)";
    if (state.session.executionProviders && state.session.executionProviders.length > 0) {
      activeProvider = state.session.executionProviders[0];
    } else if (state.session.handler && state.session.handler.key) {
      activeProvider = state.session.handler.key;
    }
    backendVal.textContent = activeProvider.toUpperCase();
    setStatus("Model Ready", `Running successfully on ${activeProvider}.`, "ready");
    startBtn.disabled = false;
  } catch (error) {
    progressBarContainer.style.display = "none";
    backendVal.textContent = "FAILED";
    setStatus("Error Loading Model", error.message || "Failed to initialize.", "error");
    console.error(error);
  }
}

// Convert image buffer to Float32 CHW planar layout [1, 3, 200, 200]
function preprocess(ctx) {
  const width = 200;
  const height = 200;
  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;
  
  const floatData = new Float32Array(3 * width * height);
  
  // ImageNet stats
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let i = 0; i < width * height; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    
    // Normalization & CHW packaging
    floatData[i] = (r / 255.0 - mean[0]) / std[0]; // Red
    floatData[width * height + i] = (g / 255.0 - mean[1]) / std[1]; // Green
    floatData[2 * width * height + i] = (b / 255.0 - mean[2]) / std[2]; // Blue
  }

  return new ort.Tensor("float32", floatData, [1, 3, width, height]);
}

// Commit logic
function commitCharacter(char) {
  const now = Date.now();
  if (now < state.commitLockUntil) return;
  
  if (char === " ") {
    if (!transcript.value.endsWith(" ") && transcript.value.length > 0) {
      transcript.value += " ";
    }
  } else {
    transcript.value += char;
  }
  
  transcript.focus();
  transcript.setSelectionRange(transcript.value.length, transcript.value.length);
  state.commitLockUntil = now + STABLE_COOLDOWN_MS;
}

function commitBackspace() {
  const now = Date.now();
  if (now < state.commitLockUntil) return;
  transcript.value = transcript.value.slice(0, -1);
  transcript.focus();
  state.commitLockUntil = now + STABLE_COOLDOWN_MS;
}

// Capture frame and evaluate using local session
async function runInference() {
  if (!state.session || state.isPredicting) return;
  if (!state.stream || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;

  state.isPredicting = true;

  try {
    // 1. Draw mirrored stream on display canvas
    const displayCtx = canvas.getContext("2d");
    displayCtx.save();
    displayCtx.translate(canvas.width, 0);
    displayCtx.scale(-1, 1);
    displayCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
    displayCtx.restore();

    // 2. Crop square from input stream (offset left/right based on active hand)
    const size = Math.min(video.videoWidth, video.videoHeight);
    const remainingWidth = video.videoWidth - size;
    let sx = remainingWidth / 2;
    if (state.hand === "left") {
      sx = remainingWidth * 0.95;
    } else if (state.hand === "right") {
      sx = remainingWidth * 0.05;
    }
    const sy = (video.videoHeight - size) / 2;

    preCtx.save();
    
    // Horizontal Mirror Flip if left hand is selected to help generalization
    if (state.hand === "left") {
      preCtx.translate(preCanvas.width, 0);
      preCtx.scale(-1, 1);
    }
    
    preCtx.drawImage(
      video,
      sx, sy, size, size,
      0, 0, preCanvas.width, preCanvas.height
    );
    preCtx.restore();

    // 3. Preprocess to ONNX Float32 tensor
    const tensor = preprocess(preCtx);

    // 4. Run session inference
    const outputMap = await state.session.run({ input: tensor });
    const outputTensor = outputMap.output || Object.values(outputMap)[0];
    const logits = outputTensor.data;

    // 5. Apply Softmax to determine confidence scores
    const probabilities = softmax(Array.from(logits));

    // 6. Map predictions to classes list
    const mappedPreds = probabilities.map((prob, idx) => ({
      label: state.classes[idx] || `Class_${idx}`,
      confidence: prob
    }));

    // Sort descending
    mappedPreds.sort((a, b) => b.confidence - a.confidence);

    const top1 = mappedPreds[0];
    const top3 = mappedPreds.slice(0, 3);

    // Update UI elements
    predictionLetter.textContent = top1.label;
    if (top1.label.length > 1) {
      predictionLetter.classList.add("word");
    } else {
      predictionLetter.classList.remove("word");
    }
    confidenceFill.style.width = `${Math.round(top1.confidence * 100)}%`;
    confidenceLabel.textContent = `${Math.round(top1.confidence * 100)}%`;

    // Render list alternatives
    topK.innerHTML = "";
    top3.forEach((item, index) => {
      const li = document.createElement("li");
      li.innerHTML = `<span>${index + 1}. ${item.label}</span><strong>${Math.round(item.confidence * 100)}%</strong>`;
      topK.appendChild(li);
    });

    trackingState.textContent = "Tracking active";
    trackingState.className = "badge-accent";
    cameraPanel.classList.add("tracking");

    // Commit filtering
    if (top1.confidence >= 0.70) {
      if (top1.label !== state.lastPrediction) {
        state.lastPrediction = top1.label;
        state.stableSince = Date.now();
      }

      const timeStable = Date.now() - state.stableSince;
      
      // Compute required hold threshold (faster commit for higher confidence)
      let requiredDuration = 1200;
      if (top1.confidence >= 0.95) requiredDuration = 450;
      else if (top1.confidence >= 0.85) requiredDuration = 700;

      if (timeStable >= requiredDuration) {
        const normalized = top1.label.toLowerCase();
        
        if (normalized === "nothing") {
          state.lastPrediction = null;
          state.stableSince = null;
        } else if (normalized === "space") {
          commitCharacter(" ");
        } else if (normalized === "del" || normalized === "backspace") {
          commitBackspace();
        } else {
          commitCharacter(top1.label);
        }
      }
    } else {
      state.lastPrediction = null;
      state.stableSince = null;
    }

  } catch (err) {
    console.error("Inference execution failed:", err);
    trackingState.textContent = "Inference error";
    trackingState.className = "status-dot error";
  } finally {
    state.isPredicting = false;
  }
}

// Camera Operations
async function startCamera() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 1280 }
      },
      audio: false
    });

    video.srcObject = state.stream;
    await video.play();

    cameraBadge.textContent = "Camera Active";
    cameraBadge.className = "badge-accent";
    startBtn.disabled = true;
    stopBtn.disabled = false;

    // Start prediction loop timer
    if (state.timer) clearInterval(state.timer);
    state.timer = setInterval(runInference, PREDICT_INTERVAL_MS);
  } catch (error) {
    console.error("Camera activation failed:", error);
    cameraBadge.textContent = "Access Denied";
    cameraBadge.style.color = "var(--danger)";
    alert("Could not access your camera. Please ensure permissions are granted.");
  }
}

function stopCamera() {
  if (state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }

  if (state.stream) {
    state.stream.getTracks().forEach(track => track.stop());
    state.stream = null;
  }

  video.srcObject = null;
  cameraBadge.textContent = "Camera Stopped";
  cameraBadge.className = "badge-outline";
  cameraPanel.classList.remove("tracking");
  
  startBtn.disabled = false;
  stopBtn.disabled = true;
  
  predictionLetter.textContent = "-";
  predictionLetter.classList.remove("word");
  confidenceFill.style.width = "0%";
  confidenceLabel.textContent = "0%";
  trackingState.textContent = "Inference Idle";
  trackingState.className = "badge-outline";
}

// Setup Event Handlers
startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);

clearBtn.addEventListener("click", () => {
  transcript.value = "";
  transcript.focus();
});

backspaceBtn.addEventListener("click", () => {
  transcript.value = transcript.value.slice(0, -1);
  transcript.focus();
});

spaceBtn.addEventListener("click", () => {
  if (!transcript.value.endsWith(" ") && transcript.value.length > 0) {
    transcript.value += " ";
  }
  transcript.focus();
});

leftHandBtn.addEventListener("click", () => {
  state.hand = "left";
  leftHandBtn.classList.add("active");
  rightHandBtn.classList.remove("active");
  cameraPanel.classList.add("hand-left");
  cameraPanel.classList.remove("hand-right");
});

rightHandBtn.addEventListener("click", () => {
  state.hand = "right";
  rightHandBtn.classList.add("active");
  leftHandBtn.classList.remove("active");
  cameraPanel.classList.add("hand-right");
  cameraPanel.classList.remove("hand-left");
});

// Run loader on startup
window.addEventListener("DOMContentLoaded", () => {
  cameraPanel.classList.add("hand-left");
  loadModel();
});
