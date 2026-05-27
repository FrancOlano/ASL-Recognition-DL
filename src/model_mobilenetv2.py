"""MobileNetV2 Transfer Learning model for ASL Fingerspelling Recognition (PyTorch)."""

import torch
import torch.nn as nn
from torchvision import models

import config


def build_mobilenet_v2(num_classes=config.NUM_CLASSES):
    """
    Build MobileNetV2 model with transfer learning for ASL classification.

    The base model is pre-trained on ImageNet with frozen feature extraction layers.
    Only the classifier head is trained.

    Args:
        num_classes: Number of output classes (24 for A-Z minus J and Z).

    Returns:
        MobileNetV2 model ready for training with frozen base layers.
    """

    # Load pre-trained MobileNetV2 from ImageNet
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze all feature extraction layers to preserve pre-trained ImageNet knowledge
    for param in model.features.parameters():
        param.requires_grad = False

    # Get the input features of the original classifier
    in_features = model.classifier[1].in_features

    # Replace classifier with custom head for ASL classification
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )

    # Display model architecture summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"MobileNetV2 (Transfer Learning) built successfully!")
    print(f"Number of classes: {num_classes}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} (classifier only)")
    print(f"Training strategy: Fine-tune classifier head (base model frozen)")

    return model
