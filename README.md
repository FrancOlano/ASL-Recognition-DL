
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

The premium client-side demo runs entirely in the browser using ONNX Runtime Web and WebGL. Because browsers restrict fetching local files (`model.onnx` and `classes.json`) via the `file://` protocol (due to CORS security rules), you need to serve the files using a simple local web server.

To run the demo, execute the included script from the project root:

```powershell
python demo/app.py
```

After starting the server, open your browser and navigate to **`http://127.0.0.1:5000`**.

Alternatively, you can run any other static web server targeting the `demo/` directory:

```powershell
python -m http.server 8000 --directory demo
```
Or use the **Live Server** extension in VS Code to serve the `demo/` folder.

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


