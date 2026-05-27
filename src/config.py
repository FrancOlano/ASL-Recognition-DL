"""Configuration file for ASL Fingerspelling Recognition Model (PyTorch)."""

import torch
from pathlib import Path

# Model Selection
# Options: "custom_cnn" or "mobilenet_v2"
MODEL_TYPE = "custom_cnn"  # Change to "mobilenet_v2" for transfer learning comparison

# Model Architecture & Training Hyperparameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 24  # A-Z excluding J and Z (which require motion)

# Hyperparameters (adjusted based on MODEL_TYPE)
if MODEL_TYPE == "custom_cnn":
    EPOCHS = 30  # More epochs needed for training from scratch
    LEARNING_RATE = 1e-3  # Higher LR for training from scratch
    OPTIMIZER_TYPE = "sgd"  # SGD with momentum works best for custom CNN
elif MODEL_TYPE == "mobilenet_v2":
    EPOCHS = 20  # Fewer epochs needed for transfer learning
    LEARNING_RATE = 1e-4  # Smaller LR for fine-tuning pretrained model
    OPTIMIZER_TYPE = "adam"  # Adam works well for fine-tuning
else:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}. Choose 'custom_cnn' or 'mobilenet_v2'")

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_DATA_DIR = DATA_DIR  # Folder structure: A/, B/, C/, ... X/, Y/
MODEL_OUTPUT_DIR = Path("/kaggle/working")
BEST_MODEL_PATH = MODEL_OUTPUT_DIR / "best_model.pth"

# Data Split
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.2

# Augmentation Parameters
ROTATION_DEGREES = 20
HORIZONTAL_FLIP_PROB = 0.5

# Optimizer Configuration (adapted based on MODEL_TYPE)
if MODEL_TYPE == "custom_cnn":
    OPTIMIZER_MOMENTUM = 0.9
    OPTIMIZER_WEIGHT_DECAY = 5e-4  # L2 regularization to prevent overfitting in custom CNN
else:  # mobilenet_v2
    OPTIMIZER_MOMENTUM = 0.9
    OPTIMIZER_WEIGHT_DECAY = 1e-5  # Smaller L2 regularization for transfer learning

# Random Seed for Reproducibility
SEED = 42

# Number of Workers for DataLoader
NUM_WORKERS = 4

# ImageNet Normalization Statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
