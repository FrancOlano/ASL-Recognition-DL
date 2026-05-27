# ASL Fingerspelling Recognition (PyTorch) 🧏

A deep learning project to recognize American Sign Language (ASL) fingerspelling in real-time. This project provides **two model architectures** for comparison:

1. **Custom CNN** - Built from scratch with PyTorch (2.5M parameters, all trainable)
2. **MobileNetV2** - Transfer learning with pre-trained ImageNet weights (14M total, 130K trainable)

Train either model individually or run both for a head-to-head comparison on static ASL letters (A-Z, excluding moving letters J and Z).

## 📁 Repository Structure

```
asl-fingerspelling-recognition/
├── data/                      # Dataset (ignored in git)
│   ├── raw/                   # Original dataset
│   └── processed/             # Preprocessed images organized by class (A/, B/, ..., Y/)
├── notebooks/                 # Jupyter/Kaggle notebooks
│   └── kaggle_training.ipynb  # Kaggle training notebook
├── src/                       # PyTorch source code
│   ├── __init__.py
│   ├── config.py              # Hyperparameters and paths
│   ├── dataset.py             # ImageFolder and DataLoader logic
│   ├── model.py               # Custom CNN architecture (ASLCustomCNN)
│   └── train.py               # Main training loop
├── models/                    # Saved .pth weights
│   └── checkpoints/
├── .gitignore                 # Git ignore rules
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## ⚙️ Environment Setup

### Local Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/asl-fingerspelling-recognition.git
   cd asl-fingerspelling-recognition
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Prepare your dataset:**
   Organize your ASL dataset in `data/processed/` with the following structure:
   ```
   data/processed/
   ├── A/
   ├── B/
   ├── C/
   ...
   ├── X/
   ├── Y/
   ```
   Each class folder should contain `.jpg` or `.png` images.

### Kaggle Setup
This project is optimized for Kaggle's free GPUs.

1. Create a new notebook on Kaggle.
2. Enable the **GPU T4 x2** accelerator in session options.
3. Add your ASL dataset via the "Add Data" button.
4. Either:
   - Clone this repo directly: `!git clone https://github.com/YOUR_USERNAME/asl-fingerspelling-recognition.git`
   - Or upload the `src/` folder manually
5. Run the Kaggle notebook cells (see `notebooks/kaggle_training.ipynb`).

## 🚀 How to Reproduce Results

### 1. Configure Hyperparameters & Select Model
Edit `src/config.py` to adjust:
- `MODEL_TYPE`: Choose between `"custom_cnn"` (default) or `"mobilenet_v2"`
- `IMAGE_SIZE`: Image dimensions (default: 224x224)
- `BATCH_SIZE`: Training batch size (default: 32)
- `EPOCHS`: Number of training epochs (auto-adjusted per model)
- `LEARNING_RATE`: Optimizer learning rate (auto-adjusted per model)
- `DATA_DIR`: Path to your dataset
- `DEVICE`: Auto-detected (CUDA/CPU)

**Model-Specific Defaults:**
- **Custom CNN**: EPOCHS=30, LR=1e-3, Optimizer=SGD+Momentum
- **MobileNetV2**: EPOCHS=20, LR=1e-4, Optimizer=Adam

### 2. Train the Model

#### Option A: Train Single Model
From the repository root, run:
```bash
python src/train.py
```

The script will use the `MODEL_TYPE` specified in `src/config.py` and train accordingly.

#### Option B: Train & Compare Both Models
To train both Custom CNN and MobileNetV2 and compare results:
```bash
python src/train_comparison.py
```

This will:
- Train Custom CNN for 30 epochs
- Train MobileNetV2 for 20 epochs
- Save both best models: `best_model_custom_cnn.pth`, `best_model_mobilenet_v2.pth`
- Generate a comparison report: `comparison_results.json`
- Print a summary table showing performance differences

### 3. Monitor Training
The training output includes:
- Per-batch loss updates (every 10 batches)
- Per-epoch training and validation metrics
- Automatic model checkpointing when validation accuracy improves
- Model selection information

Example output:
```
Epoch [1/30]
  Batch [10/100], Loss: 2.4532
  Batch [20/100], Loss: 1.8234
Training   - Loss: 1.5234, Accuracy: 0.6234
Validation - Loss: 1.2123, Accuracy: 0.7123
✓ Model checkpoint saved to /kaggle/working/best_model.pth
  Best validation accuracy: 0.7123 (Epoch 1)
```

## 🔬 Model Comparison: Custom CNN vs. MobileNetV2

| Aspect | Custom CNN | MobileNetV2 |
|--------|-----------|------------|
| **Architecture** | 4 Conv Blocks (3→32→64→128→256) | Inverted residual blocks |
| **Total Parameters** | 2.5M | 14M |
| **Trainable Parameters** | 2.5M (100%) | 130K (0.9%) |
| **Training Strategy** | All parameters from scratch | Transfer learning (frozen base) |
| **Optimizer** | SGD + Momentum | Adam |
| **Learning Rate** | 1e-3 | 1e-4 |
| **Training Epochs** | 30 | 20 |
| **Weight Decay** | 5e-4 | 1e-5 |
| **Training Time** | 30-50 min (T4) | 15-30 min (T4) |
| **Model Size** | ~9.5 MB | ~13 MB |
| **Inference Speed** | Fast | Very Fast |
| **Expected Val Acc** | 80-88% | 85-92% |
| **Use Case** | Research, customization | Production, real-time |

### When to Use Each Model

**Choose Custom CNN if:**
- You want full control over architecture
- You have plenty of training data
- You want to understand deep learning fundamentals
- You need to experiment with custom layers

**Choose MobileNetV2 if:**
- You want faster convergence
- You have limited training data
- You need real-time inference speed
- You want better accuracy with less training time

### Architecture Details
```
Input (B, 3, 224, 224)
    ↓
Conv Block 1: Conv(3→32) + BatchNorm + ReLU + MaxPool(2)
    ↓ (B, 32, 112, 112)
Conv Block 2: Conv(32→64) + BatchNorm + ReLU + MaxPool(2)
    ↓ (B, 64, 56, 56)
Conv Block 3: Conv(64→128) + BatchNorm + ReLU + MaxPool(2)
    ↓ (B, 128, 28, 28)
Conv Block 4: Conv(128→256) + BatchNorm + ReLU + MaxPool(2)
    ↓ (B, 256, 14, 14)
AdaptiveAvgPool2d((1, 1))
    ↓ (B, 256, 1, 1)
Flatten
    ↓ (B, 256)
Dropout (p=0.5)
    ↓
Linear (256 → 24 classes)
    ↓
Output: (B, 24) logits
```

### Why a Custom CNN from Scratch?
- **Full Control:** Every layer is customized for the ASL fingerspelling task
- **Lightweight:** ~2.5M parameters vs. 14M for MobileNetV2
- **Efficient Training:** Trains faster on limited GPU memory
- **Research-Oriented:** Great for experimentation and understanding deep learning fundamentals
- **Batch Normalization:** Stabilizes training and accelerates convergence
- **Kaiming He Initialization:** Optimal weight initialization for ReLU networks

### Training Strategy
- **All Parameters Trained:** Unlike transfer learning, all 2.5M parameters are updated during training
- **Regularization:** Dropout(0.5) + L2 weight decay (5e-4) prevent overfitting
- **Data Augmentation:** Critical for learning diverse hand positions and lighting conditions
- **Optimizer:** SGD with momentum (0.9) and learning rate 1e-3
- **Loss:** CrossEntropyLoss for multi-class classification
- **Training Epochs:** 30 epochs (compared to 20 for transfer learning models)

## 📊 Data Augmentation

**Data augmentation is CRITICAL when training a custom CNN from scratch** to prevent overfitting and improve generalization across different hand positions, lighting conditions, and backgrounds.

### Training Set Transformations
- **Resize:** 256×256
- **RandomCrop:** 224×224 (introduces spatial variation)
- **RandomHorizontalFlip:** 50% probability (mirrors hand positions)
- **RandomRotation:** ±20 degrees (handles rotated signs)
- **Normalization:** ImageNet mean/std

### Validation Set Transformations
- **Resize:** 256×256
- **CenterCrop:** 224×224 (consistent center positioning)
- **Normalization:** ImageNet mean/std
(No augmentation for consistent validation metrics)

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0.0 | Deep learning framework |
| `torchvision` | ≥0.15.0 | Vision models & transforms |
| `numpy` | ≥1.21.0 | Numerical computing |
| `matplotlib` | ≥3.4.0 | Visualization |
| `Pillow` | ≥9.0.0 | Image processing |
| `tqdm` | ≥4.62.0 | Progress bars |
| `kaggle` | ≥1.5.12 | Kaggle API integration |

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 🔍 Expected Performance

On typical ASL datasets with ~2,000-5,000 images per class:
- **Training Accuracy:** 90-95%
- **Validation Accuracy:** 80-88%
- **Training Time:** 30-50 minutes on GPU (T4 x2 on Kaggle)
- **Model Size:** ~9.5 MB (state_dict)

**Note:** Custom CNNs typically require more epochs (30 vs. 20) and stronger augmentation compared to transfer learning models, but offer more flexibility for domain-specific optimization.

## 💾 Model Checkpointing

The training script automatically saves the best model based on validation accuracy:
```python
# Best model is saved to:
/kaggle/working/best_model.pth
```

To load the best model for inference:
```python
import torch
from src.model import build_model

model = build_model(num_classes=24)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Now use model for inference
```

## 🎯 Next Steps

After training:
1. **Evaluate on Test Set:** Create `src/evaluate.py` to compute per-class metrics
2. **Inference Pipeline:** Build `src/inference.py` for real-time prediction on webcam
3. **Visualization:** Add confusion matrix and per-class accuracy analysis
4. **Deployment:** Export to ONNX or TorchScript for production

## 📝 License

This project is licensed under the MIT License. See LICENSE for details.

## 🙏 Acknowledgments

- **Custom CNN Architecture:** Inspired by VGG and ResNet design principles
- **PyTorch Documentation:** [PyTorch Tutorials](https://pytorch.org/tutorials/)
- **Batch Normalization:** [Batch Normalization Paper](https://arxiv.org/abs/1502.03167)
- **Kaiming He Initialization:** [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852)
- **ASL Dataset:** Ensure proper attribution if using public datasets

## 📚 Documentation

- **[README.md](README.md)** - Project overview and quick start
- **[MODEL_COMPARISON_GUIDE.md](MODEL_COMPARISON_GUIDE.md)** - Detailed guide for model selection and comparison
- **[src/config.py](src/config.py)** - Configuration and hyperparameter settings

---

**Happy training! 🚀**