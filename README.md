# ASL-Recognition-DL

American Sign Language (ASL) fingerspelling recognition using PyTorch. This repository contains training utilities, model architectures, a demo app, and example notebooks used in development.

## 1. Reproducing Results (Environment & Dependencies)

To reproduce the results, you need to set up the environment and prepare the data.

### Environment Setup

Create and activate a virtual environment (Windows example):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

Install the required Python packages:
```powershell
pip install -r requirements.txt
```

### Data Preparation
The dataset should be prepared under the `data/processed/` directory. Each class should have its own folder (e.g., `data/processed/A/`, `data/processed/B/`, etc.). 

## 2. Experimentation & Testing

The repository provides several tools to experiment with different architectures, hyperparameters, and to test your models.

### Training
Training scripts and utilities are located in `engine/`. You can adjust paths, model types (`ASLCustomCNN`, `ASLInceptionV3`, `ASLMobileNetV2`), and hyperparameters in `engine/config.py`.

Run the training pipeline from the project root:
```powershell
python -m engine.train
```
During training, the best checkpoints will be saved to `checkpoints/` and training metrics/history will be logged to `results/`.

### Model Architectures
Custom architectures and wrappers are located in the `models/` directory. You can extend this module with new architectures and register them in `engine/model_factory.py`.

### Testing and Evaluation
We provide plotting scripts in the `engine/` directory to evaluate your models visually:
- **Confusion Matrix:** `python -m engine.plot_confusion_matrix`
- **Training Curves:** `python -m engine.plot_training_curves`

### 🌟 Notebooks & Kaggle Environment

For an interactive, zero-setup approach to training and experimentation, we provide several ready-to-use Jupyter Notebooks in the `notebooks/` directory.

#### Available Notebooks

- **`kaggle_training.ipynb`**: The primary training notebook. It covers exploratory data analysis (EDA), configuring data loaders, and executing the PyTorch training pipeline from scratch.
- **`inference_simple.ipynb`**: A lightweight notebook to test inference. It loads a trained checkpoint from your local directory and predicts the class of a single image.
- **`testing_metrics.ipynb`**: Evaluates model performance across the base model variants (26 classes). It computes and outputs F1 scores, accuracy, loss, and confusion matrices.
- **`testing_metrics_29.ipynb`**: Similar to the above, but specifically tailored to evaluate the fine-tuned 29-class model.
- **`transfer_learning_new_symbols.ipynb`**: Demonstrates transfer learning by taking a pretrained 26-class MobileNetV2 model and fine-tuning it to recognize 29 classes (adding `del`, `nothing`, and `space`).

#### Using Kaggle for Training
The notebooks (especially `kaggle_training.ipynb` and `transfer_learning_new_symbols.ipynb`) are designed to be run directly on [Kaggle](https://www.kaggle.com/) to easily leverage free GPUs.

To use them in Kaggle:
1. Create a new notebook on Kaggle and import the desired notebook file (File > Import Notebook).
2. Attach the required dataset to your Kaggle environment by adding it via "Add Data" in the right-hand sidebar.
3. Turn on the **GPU accelerator** in the notebook settings to significantly speed up training or evaluation.
4. Run the notebook cells sequentially.

*Note: Checkpoints and models saved during a Kaggle session can be downloaded and placed in your local `checkpoints/` folder to run the local demo application.*

## 3. Demo Application

The client-side demo runs entirely in the browser using ONNX Runtime Web and WebGL. Because browsers restrict fetching local files (`model.onnx` and `classes.json`) via the `file://` protocol (due to CORS security rules), you need to serve the files using a simple local web server.

### Exporting your Model
First, export your trained PyTorch checkpoint to ONNX format (adjust paths within the script if necessary):
```powershell
python -m engine.export_onnx
```

### Running the Demo
Execute the included script from the project root to start the Flask server:
```powershell
python demo/app.py
```

After starting the server, open your browser and navigate to **`http://127.0.0.1:5000`**.

Alternatively, you can run any other static web server targeting the `demo/` directory:
```powershell
python -m http.server 8000 --directory demo
```
Or use the **Live Server** extension in VS Code to serve the `demo/` folder.

---

### Acknowledgements

- **ASL Alphabet Dataset**: The primary dataset used to train the models in this repository. 
  - Akash Nagaraj. (2018). *ASL Alphabet* [Dataset]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/29550
