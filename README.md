
ASL-Recognition-DL
==================

American Sign Language (ASL) fingerspelling recognition using PyTorch. This repository contains training utilities, model architectures, a demo app, and example notebooks used in development.

Repository layout
-----------------

- `context/` — project context and helper configuration
- `data/` — dataset files
    - `raw/` — original/raw files
    - `processed/` — preprocessed images organized by class (A/, B/, ...)
- `demo/` — Flask demo app and static assets (`demo/app.py`)
    - `static/`, `templates/`
- `engine/` — training utilities and dataset loaders
    - `dataset.py`, `train.py`, `model_factory.py`, `config.py`
- `models/` — model architecture source files
    - `ASLCustomCNN.py`, `ASLInceptionV3.py`, `ASLMobileNetV2.py`
- `checkpoints/` — saved model weights (e.g. `preprocess.pth`)
- `notebooks/` — Jupyter notebooks (e.g. `kaggle_training.ipynb`)
- `requirements.txt` — Python dependencies

Quick start
-----------

1. Create and activate a virtual environment (Windows example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the Demo Locally
--------------------

### Option 1: Static Web Demo (Recommended)
The premium client-side demo runs entirely in the browser using ONNX Runtime Web and WebGL. Because browsers restrict fetching local files (`model.onnx` and `classes.json`) via the `file://` protocol (due to CORS security rules), you need to serve the files using a simple local web server.

#### Using Python (Built-in)
If you have Python installed, run this single command in your terminal from the project root:

```powershell
python -m http.server 8000 --directory docs
```

#### Using Node.js / npm
If you have Node.js installed, you can run:

```powershell
npx serve docs
```

#### Using VS Code
If you use VS Code, you can install the **Live Server** extension, open the `docs/` folder, and click "Go Live" at the bottom right.

After starting the server, open your browser and navigate to:
**`http://localhost:8000`** (or the port specified by your server).

---

### Option 2: Flask Python Demo (Legacy)
If you prefer running the original Python/Flask backend demo:

```powershell
python ./demo/app.py
```

This demo expects model checkpoints to be in the `checkpoints/` folder.

Training
--------

Training scripts and utilities live in `engine/`. Prepare your dataset under `data/processed/` (one folder per class) and run:

```powershell
python -m engine.train
```

Adjust paths and hyperparameters in `engine/config.py`.

Models & checkpoints
--------------------

- Architectures: `models/` (see `ASLCustomCNN.py`, `ASLInceptionV3.py`, `ASLMobileNetV2.py`).
- Checkpoints: `checkpoints/`.

Notebooks
---------

See `notebooks/kaggle_training.ipynb` for experiments and data exploration.

Contributing
------------

Please open issues or PRs. For reproducible runs include environment details and the checkpoint/config used.

Contact
-------

Maintainer: refer to repository metadata or the original project owner.


