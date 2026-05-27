"""
Configuration file for ASL Fingerspelling Recognition Model (PyTorch).
Generalized for Phase 1 (Comparison) and Phase 2 (Optimization).
"""

import torch
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. Model Selection & Strategy
# -----------------------------------------------------------------------------
# Options: "custom_cnn", "mobilenet_v2", "inception_v3"
MODEL_TYPE = "custom_cnn" 

# Strategy: True for Feature Extraction (frozen base), False for Training from scratch
# Note: "custom_cnn" ignores this flag as it is always trained from scratch.
PRETRAINED = True 

# -----------------------------------------------------------------------------
# 2. Dataset & Architecture Parameters
# -----------------------------------------------------------------------------
# Strictly 200x200 as per project requirements (no scaling/cropping)
IMAGE_SIZE = 200 

# Initial alphabet training (26 classes). Update to 29 for the later demo phase.
NUM_CLASSES = 26 

# Hardware constraints / Training setup
BATCH_SIZE = 10 

# -----------------------------------------------------------------------------
# 3. Training Hyperparameters
# -----------------------------------------------------------------------------
# Testing ranges defined in proposal: 10 to 30 epochs
EPOCHS = 20 

# Optimizer selection for Phase 1 Controlled Comparison
OPTIMIZER = "sgd"  # Options: "sgd" or "adam"

# Dynamic hyperparameters based on training strategy
if MODEL_TYPE == "custom_cnn" or not PRETRAINED:
    LEARNING_RATE = 1e-3
    OPTIMIZER_WEIGHT_DECAY = 5e-4  # Stronger L2 regularization for training from scratch
else:
    LEARNING_RATE = 1e-4
    OPTIMIZER_WEIGHT_DECAY = 1e-5  # Smaller L2 regularization for fine-tuning

OPTIMIZER_MOMENTUM = 0.9  # Used if OPTIMIZER == "sgd"

# -----------------------------------------------------------------------------
# 4. Data Augmentation (Safe Augmentations Only)
# -----------------------------------------------------------------------------
# DANGEROUS: Horizontal flips and large rotations are strictly omitted!
ROTATION_DEGREES = 15          # Small rotations (±10-15°) to simulate natural hand variation
TRANSLATION_FRACTION = 0.1     # Random slight translations for hand position robustness
JITTER_BRIGHTNESS = 0.2        # Brightness changes to handle lighting variability
JITTER_CONTRAST = 0.2          # Contrast changes to handle lighting variability

# -----------------------------------------------------------------------------
# 5. Device & Reproducibility
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
NUM_WORKERS = 2  # Adjusted for stable data loading with smaller batch sizes

# ImageNet Normalization Statistics (Required for MobileNetV2 and InceptionV3)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# -----------------------------------------------------------------------------
# 6. Paths & Directories
# -----------------------------------------------------------------------------
# Environment detection (automatically handles Kaggle vs. Local execution)
KAGGLE = Path("/kaggle/working").exists()

if KAGGLE:
    PROJECT_ROOT = Path("/kaggle/working")
    # Update "YOUR_DATASET_NAME" to match your exact Kaggle dataset import path
    DATA_DIR = Path("/kaggle/input/YOUR_DATASET_NAME") 
else:
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_OUTPUT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

# Dynamic checkpoint naming based on the active configuration
model_suffix = "pretrained" if PRETRAINED and MODEL_TYPE != "custom_cnn" else "scratch"
BEST_MODEL_PATH = MODEL_OUTPUT_DIR / f"best_{MODEL_TYPE}_{model_suffix}.pth"

# Data Split Strategy
TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15