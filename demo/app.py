from __future__ import annotations

import base64
import io
import json
import sys
from threading import Lock
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import config as train_config
from models.ASLMobileNetV2 import ASLMobileNetV2
from models.ASLInceptionV3 import ASLInceptionV3


app = Flask(__name__)

MODEL_REGISTRY = {
    "mobilenet_v2_scratch": {
        "label": "MobileNetV2 scratch",
        "architecture": "mobilenet_v2",
        "checkpoint": CHECKPOINT_DIR / "best_mobilenet_v2_scratch.pth",
    },
    "mobilenet_v2_finetuned_29": {
        "label": "MobileNetV2 finetuned (29 classes)",
        "architecture": "mobilenet_v2",
        "checkpoint": CHECKPOINT_DIR / "best_mobilenet_v2_finetuned_29.pth",
    },
    "inception_v3_scratch": {
        "label": "InceptionV3 scratch",
        "architecture": "inception_v3",
        "checkpoint": CHECKPOINT_DIR / "best_inception_v3_scratch.pth",
    },
}
DEFAULT_MODEL_KEY = "mobilenet_v2_finetuned_29"

DEFAULT_CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space",
]

PREPROCESS = transforms.Compose(
    [
        transforms.Resize((train_config.IMAGE_SIZE, train_config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_config.IMAGENET_MEAN, std=train_config.IMAGENET_STD),
    ]
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
MODEL_LOCK = Lock()
MODEL_INFO = {
    "ready": False,
    "key": None,
    "label": None,
    "architecture": None,
    "checkpoint": None,
    "message": "",
    "class_names": DEFAULT_CLASS_NAMES,
}


def _get_class_names(num_classes: int) -> list[str]:
    for filename in (f"classes_{num_classes}.json", "classes.json"):
        classes_path = RESULTS_DIR / filename
        if classes_path.is_file():
            try:
                payload = json.loads(classes_path.read_text(encoding="utf-8"))
                if isinstance(payload, list) and len(payload) == num_classes and all(isinstance(item, str) for item in payload):
                    return payload
            except Exception:
                pass
    return DEFAULT_CLASS_NAMES[:num_classes]


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


def _available_model_items() -> list[tuple[str, dict]]:
    items = []
    for key, model_config in MODEL_REGISTRY.items():
        if model_config["checkpoint"].is_file():
            items.append((key, model_config))
    return items


def _model_payload(key: str, model_config: dict, *, ready: bool, message: str = "", class_names: list[str] = None):
    checkpoint = model_config["checkpoint"]
    return {
        "ready": ready,
        "key": key,
        "label": model_config["label"],
        "architecture": model_config["architecture"],
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)) if checkpoint.exists() else str(checkpoint),
        "message": message,
        "class_names": class_names or DEFAULT_CLASS_NAMES,
    }


def _build_model(architecture: str, num_classes: int):
    if architecture == "mobilenet_v2":
        return ASLMobileNetV2(num_classes=num_classes, pretrained=False)
    if architecture == "inception_v3":
        return ASLInceptionV3(num_classes=num_classes, pretrained=False)
    raise ValueError(f"Unknown architecture: {architecture}")


def _load_model_by_key(model_key: str) -> dict:
    global MODEL, MODEL_INFO

    model_config = MODEL_REGISTRY.get(model_key)
    if model_config is None:
        MODEL = None
        MODEL_INFO = {
            "ready": False,
            "key": model_key,
            "label": model_key,
            "architecture": None,
            "checkpoint": None,
            "message": f"Unknown model key: {model_key}",
            "class_names": DEFAULT_CLASS_NAMES,
        }
        return MODEL_INFO

    checkpoint_path = model_config["checkpoint"]
    if not checkpoint_path.is_file():
        MODEL = None
        MODEL_INFO = _model_payload(
            model_key,
            model_config,
            ready=False,
            message=f"Checkpoint not found: {checkpoint_path}",
        )
        return MODEL_INFO

    raw_checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = _strip_module_prefix(_extract_state_dict(raw_checkpoint))

    num_classes = len(DEFAULT_CLASS_NAMES)
    for key in ("model.classifier.1.weight", "model.fc.weight", "classifier.1.weight"):
        if key in state_dict:
            num_classes = state_dict[key].shape[0]
            break
            
    class_names = _get_class_names(num_classes)

    try:
        model = _build_model(model_config["architecture"], num_classes=num_classes).to(DEVICE)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        MODEL = model
        MODEL_INFO = _model_payload(model_key, model_config, ready=True, class_names=class_names)
    except Exception as exc:
        MODEL = None
        MODEL_INFO = _model_payload(
            model_key,
            model_config,
            ready=False,
            message=f"Failed to load checkpoint: {exc}",
            class_names=class_names,
        )

    return MODEL_INFO


def _load_default_model() -> dict:
    available_models = _available_model_items()
    if any(key == DEFAULT_MODEL_KEY for key, _ in available_models):
        return _load_model_by_key(DEFAULT_MODEL_KEY)

    if available_models:
        return _load_model_by_key(available_models[0][0])

    global MODEL, MODEL_INFO
    MODEL = None
    MODEL_INFO = {
        "ready": False,
        "key": None,
        "label": None,
        "architecture": None,
        "checkpoint": None,
        "message": "No checkpoints found in models/checkpoints.",
        "class_names": DEFAULT_CLASS_NAMES,
    }
    return MODEL_INFO


def _predict_image_payload(image_payload: str):
    """
    Decodes, predicts, and completely destroys image variables in one scoped function
    to prevent RAM leak pileups.
    """
    if "," in image_payload:
        image_payload = image_payload.split(",", 1)[1]

    image_bytes = base64.b64decode(image_payload)

    # Use context managers to auto-close buffers and strictly manage the PIL image lifecycle
    with io.BytesIO(image_bytes) as buf:
        with Image.open(buf) as img:
            image = img.convert("RGB")

    with MODEL_LOCK:
        model = MODEL
        model_info = dict(MODEL_INFO)

    if model is None:
        image.close()
        raise RuntimeError(model_info.get("message") or "Model is not ready")

    class_names = model_info.get("class_names", DEFAULT_CLASS_NAMES)

    # Process and move tensor to Device
    tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)
    
    # 💥 CRITICAL: Explicitly destroy the PIL image from RAM immediately
    image.close()
    del image
    del image_bytes 

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        top_k = min(3, len(class_names))
        top_probabilities, top_indices = torch.topk(probabilities, k=top_k)

    top_index = int(top_indices[0].item())
    prediction = class_names[top_index]

    result = {
        "prediction": prediction,
        "confidence": float(top_probabilities[0].item()),
        "top3": [
            {
                "letter": class_names[int(index.item())],
                "confidence": float(prob.item()),
            }
            for index, prob in zip(top_indices, top_probabilities)
        ],
    }

    # 💥 CRITICAL: Explicitly clear the tensors from GPU/CPU memory before returning
    del tensor, logits, probabilities, top_probabilities, top_indices

    return result


@app.route("/")
def index():
    with MODEL_LOCK:
        model_info = dict(MODEL_INFO)

    available_models = [
        {
            "key": key,
            "label": model_config["label"],
            "architecture": model_config["architecture"],
            "checkpoint": str(model_config["checkpoint"].relative_to(PROJECT_ROOT)),
        }
        for key, model_config in _available_model_items()
    ]

    return render_template(
        "index.html",
        model_ready=model_info["ready"],
        current_model_key=model_info["key"],
        current_model_label=model_info["label"],
        model_architecture=model_info["architecture"],
        checkpoint_path=model_info["checkpoint"],
        status_message=model_info["message"],
        class_count=len(model_info.get("class_names", DEFAULT_CLASS_NAMES)),
        available_models=available_models,
        default_model_key=DEFAULT_MODEL_KEY,
    )


@app.route("/predict", methods=["POST"])
def predict():
    with MODEL_LOCK:
        if MODEL is None:
            return jsonify({"ok": False, "error": MODEL_INFO["message"]}), 503
        model_info = dict(MODEL_INFO)

    payload = request.get_json(silent=True) or {}
    image_payload = payload.get("image")
    if not image_payload:
        return jsonify({"ok": False, "error": "Missing image payload"}), 400

    try:
        prediction = _predict_image_payload(image_payload)
        return jsonify(
            {
                "ok": True,
                "model": model_info,
                "prediction": prediction["prediction"],
                "confidence": prediction["confidence"],
                "top3": prediction["top3"],
                "class_count": len(model_info.get("class_names", DEFAULT_CLASS_NAMES)),
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/status")
def status():
    return jsonify(MODEL_INFO)


@app.route("/api/models")
def models():
    return jsonify(
        {
            "current": MODEL_INFO,
            "available": [
                {
                    "key": key,
                    "label": model_config["label"],
                    "architecture": model_config["architecture"],
                    "checkpoint": str(model_config["checkpoint"].relative_to(PROJECT_ROOT)),
                }
                for key, model_config in _available_model_items()
            ],
            "default": DEFAULT_MODEL_KEY,
        }
    )


@app.route("/api/model", methods=["POST"])
def switch_model():
    payload = request.get_json(silent=True) or {}
    model_key = payload.get("model") or DEFAULT_MODEL_KEY

    with MODEL_LOCK:
        model_info = _load_model_by_key(model_key)

    status_code = 200 if model_info["ready"] else 400
    return jsonify({"ok": model_info["ready"], "model": model_info}), status_code


if __name__ == "__main__":
    with MODEL_LOCK:
        _load_default_model()
    app.run(host="127.0.0.1", port=5000, debug=True)