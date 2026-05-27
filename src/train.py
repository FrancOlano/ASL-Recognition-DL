"""Main training script for ASL Fingerspelling Recognition (PyTorch)."""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

import config
from dataset import get_data_loaders
from model import build_model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer instance.
        device: Device to run training on ('cuda' or 'cpu').

    Returns:
        Tuple of (avg_loss, accuracy) for the epoch.
    """
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy


def validate(model, val_loader, criterion, device):
    """
    Validate the model on the validation set.

    Args:
        model: PyTorch model to validate.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run validation on ('cuda' or 'cpu').

    Returns:
        Tuple of (avg_loss, accuracy) for the validation set.
    """
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Track metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    epoch_loss = running_loss / len(val_loader)
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy


def save_checkpoint(model, epoch, best_val_accuracy, checkpoint_path):
    """
    Save model checkpoint when validation accuracy improves.

    Args:
        model: Model to save.
        epoch: Current epoch number.
        best_val_accuracy: Best validation accuracy achieved.
        checkpoint_path: Path to save the checkpoint.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"✓ Model checkpoint saved to {checkpoint_path}")
    print(f"  Best validation accuracy: {best_val_accuracy:.4f} (Epoch {epoch + 1})")


def main():
    """Main training pipeline."""

    # Set random seeds for reproducibility
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)

    print(f"Device: {config.DEVICE}")
    print(f"Dataset directory: {config.DATA_DIR}")
    print(f"Model output directory: {config.MODEL_OUTPUT_DIR}\n")

    # Create output directory
    config.MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading dataset...")
    train_loader, val_loader, train_dataset, val_dataset = get_data_loaders(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    print()

    # Build model
    print("Building model...")
    model = build_model(num_classes=config.NUM_CLASSES)
    model = model.to(config.DEVICE)
    print()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Select optimizer and training parameters based on model type
    if config.MODEL_TYPE == "custom_cnn":
        # Train all parameters of the custom CNN from scratch
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.OPTIMIZER_MOMENTUM,
            weight_decay=config.OPTIMIZER_WEIGHT_DECAY
        )
        print(f"Optimizer: SGD with Momentum")
        print(f"Training strategy: Custom CNN trained from scratch (all parameters)")

    else:  # mobilenet_v2
        # Fine-tune only classifier parameters for transfer learning
        optimizer = optim.Adam(
            model.classifier.parameters(),
            lr=config.LEARNING_RATE
        )
        print(f"Optimizer: Adam")
        print(f"Training strategy: MobileNetV2 transfer learning (classifier only)")

    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Weight decay: {config.OPTIMIZER_WEIGHT_DECAY}")
    print(f"Loss function: CrossEntropyLoss")
    print(f"Epochs: {config.EPOCHS}\n")

    # Training loop
    best_val_accuracy = 0.0
    print("Starting training...\n")

    for epoch in range(config.EPOCHS):
        print(f"Epoch [{epoch + 1}/{config.EPOCHS}]")

        # Train phase
        train_loss, train_accuracy = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE
        )

        # Validation phase
        val_loss, val_accuracy = validate(
            model, val_loader, criterion, config.DEVICE
        )

        # Print epoch statistics
        print(f"Training   - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        # Save best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(model, epoch, best_val_accuracy, config.BEST_MODEL_PATH)

        print()

    print("Training completed!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Best model saved to: {config.BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
