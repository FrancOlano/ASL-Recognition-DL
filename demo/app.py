from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config as train_config
from model import ASLCustomCNN
from model_mobilenetv2 import build_mobilenet_v2


app = Flask(__name__)

CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y",
]

PREPROCESS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(train_config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_config.IMAGENET_MEAN, std=train_config.IMAGENET_STD),
    ]
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
MODEL_INFO = {
    "ready": False,
    "architecture": None,
    "checkpoint": None,
    "message": "",
}


def _find_checkpoint_path() -> Path | None:
    candidates = [
        *CHECKPOINT_DIR.glob("*.pth"),
        *CHECKPOINT_DIR.glob("*.pt"),
        *CHECKPOINT_DIR.glob("*.ckpt"),
    ]
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        return None

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _extract_state_dict(raw_checkpoint):
    if isinstance(raw_checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            value = raw_checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return raw_checkpoint


def _strip_module_prefix(state_dict):
    stripped = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            stripped[key[len("module."):]] = value
        else:
            stripped[key] = value
    return stripped


def _looks_like_mobilenet(state_dict) -> bool:
    keys = list(state_dict.keys())
    return any(key.startswith("features.") for key in keys) or any(
        key.startswith("classifier.1.") or key == "classifier.1.weight" for key in keys
    )


def _build_model(architecture: str):
    if architecture == "mobilenet_v2":
        return build_mobilenet_v2(num_classes=train_config.NUM_CLASSES)
    return ASLCustomCNN(num_classes=train_config.NUM_CLASSES)


def _load_model() -> None:
    global MODEL, MODEL_INFO

    checkpoint_path = _find_checkpoint_path()
    if checkpoint_path is None:
        MODEL_INFO = {
            "ready": False,
            "architecture": None,
            "checkpoint": None,
            "message": f"No checkpoint found in {CHECKPOINT_DIR}",
        }
        return

    raw_checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = _strip_module_prefix(_extract_state_dict(raw_checkpoint))

    architectures = ["mobilenet_v2", "custom_cnn"] if _looks_like_mobilenet(state_dict) else ["custom_cnn", "mobilenet_v2"]

    last_error = None
    for architecture in architectures:
        model = _build_model(architecture).to(DEVICE)
        try:
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            MODEL = model
            MODEL_INFO = {
                "ready": True,
                "architecture": architecture,
                "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
                "message": "",
            }
            return
        except Exception as exc:  # pragma: no cover - surfaced in app status
            last_error = exc

    MODEL = None
    MODEL_INFO = {
        "ready": False,
        "architecture": None,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "message": f"Failed to load checkpoint: {last_error}",
    }


def _decode_image(image_payload: str) -> Image.Image:
    if "," in image_payload:
        image_payload = image_payload.split(",", 1)[1]

    image_bytes = base64.b64decode(image_payload)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image


def _predict_image(image: Image.Image):
    if MODEL is None:
        raise RuntimeError("Model is not ready")

    tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        top_probabilities, top_indices = torch.topk(probabilities, k=3)

    top_index = int(top_indices[0].item())
    prediction = CLASS_NAMES[top_index]

    return {
        "prediction": prediction,
        "confidence": float(top_probabilities[0].item()),
        "top3": [
            {
                "letter": CLASS_NAMES[int(index.item())],
                "confidence": float(prob.item()),
            }
            for index, prob in zip(top_indices, top_probabilities)
        ],
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        model_ready=MODEL_INFO["ready"],
        model_architecture=MODEL_INFO["architecture"],
        checkpoint_path=MODEL_INFO["checkpoint"],
        status_message=MODEL_INFO["message"],
        class_count=len(CLASS_NAMES),
    )


@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"ok": False, "error": MODEL_INFO["message"]}), 503

    payload = request.get_json(silent=True) or {}
    image_payload = payload.get("image")
    if not image_payload:
        return jsonify({"ok": False, "error": "Missing image payload"}), 400

    try:
        image = _decode_image(image_payload)
        prediction = _predict_image(image)
        return jsonify(
            {
                "ok": True,
                "model": MODEL_INFO,
                "prediction": prediction["prediction"],
                "confidence": prediction["confidence"],
                "top3": prediction["top3"],
                "class_count": len(CLASS_NAMES),
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/status")
def status():
    return jsonify(MODEL_INFO)


if __name__ == "__main__":
    _load_model()
    app.run(host="127.0.0.1", port=5000, debug=True)