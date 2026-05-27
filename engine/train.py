"""Main generalized training script for ASL Fingerspelling Recognition (PyTorch)."""

import copy
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from . import config
except ImportError:
    import config

try:
    from .dataset import get_data_loaders
except ImportError:
    from dataset import get_data_loaders

from engine.model_factory import build_model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    return running_loss / len(train_loader), correct_predictions / total_samples


def validate(model, val_loader, criterion, device):
    """
    Validate the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    return running_loss / len(val_loader), correct_predictions / total_samples


def get_optimizer(model, optimizer_name, learning_rate, momentum, weight_decay):
    """
    Dynamically returns the optimizer, ensuring only trainable parameters are passed.
    """
    # Filter out frozen parameters (crucial for feature extraction mode)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    if optimizer_name.lower() == "adam":
        return optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    else: # Default to SGD
        return optim.SGD(trainable_params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model on a held-out dataset.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    return running_loss / len(data_loader), correct_predictions / total_samples


def train_model(
    model_type,
    pretrained,
    device,
    epochs=None,
    learning_rate=None,
    optimizer_name=None,
    momentum=None,
    weight_decay=None,
):
    """
    Main orchestrator for training a model and tracking its history.
    """
    print(f"\n{'='*70}")
    print(f"STARTING TRAINING PIPELINE: {model_type.upper()}")
    print(f"{'='*70}\n")

    # 1. Build Model
    model = build_model(
        model_type=model_type, 
        num_classes=config.NUM_CLASSES, 
        pretrained=pretrained
    )
    model = model.to(device)

    # 2. Setup Data
    print("Loading dataset...")
    train_loader, val_loader, test_loader, _, _, _ = get_data_loaders(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )

    # 3. Setup Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_name = optimizer_name or getattr(config, 'OPTIMIZER', 'SGD') # Fallback to SGD if not in config
    optimizer = get_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate or config.LEARNING_RATE,
        momentum=momentum if momentum is not None else getattr(config, 'OPTIMIZER_MOMENTUM', 0.9),
        weight_decay=weight_decay if weight_decay is not None else getattr(config, 'OPTIMIZER_WEIGHT_DECAY', 1e-4)
    )

    print(f"\nOptimizer: {optimizer_name.upper()}")
    print(f"Learning Rate: {learning_rate or config.LEARNING_RATE}")
    print(f"Epochs: {epochs or config.EPOCHS}\n")

    # 4. Initialize History Tracking
    best_val_accuracy = 0.0
    history = {
        "model_type": model_type,
        "pretrained": pretrained,
        "optimizer": optimizer_name,
        "learning_rate": learning_rate or config.LEARNING_RATE,
        "batch_size": config.BATCH_SIZE,
        "epochs": epochs or config.EPOCHS,
        "train_losses": [],
        "train_accuracies": [],
        "val_losses": [],
        "val_accuracies": [],
        "test_loss": None,
        "test_accuracy": None,
        "best_val_accuracy": 0.0,
        "best_epoch": 0
    }

    # 5. Training Loop
    checkpoint_dir = Path("/kaggle/working/checkpoints") if getattr(config, 'KAGGLE', False) else Path("./checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_suffix = "pretrained" if pretrained else "scratch"
    checkpoint_path = checkpoint_dir / f"best_{model_type}_{model_suffix}.pth"

    total_epochs = epochs or config.EPOCHS

    for epoch in range(total_epochs):
        print(f"Epoch [{epoch + 1}/{total_epochs}]")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update History
        history["train_losses"].append(train_loss)
        history["train_accuracies"].append(train_acc)
        history["val_losses"].append(val_loss)
        history["val_accuracies"].append(val_acc)

        print(f"Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # Checkpoint saving
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            history["best_val_accuracy"] = best_val_accuracy
            history["best_epoch"] = epoch + 1
            
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path.name} (Val Acc: {best_val_accuracy:.4f})")
        print()

    # 6. Evaluate on test data with the best checkpoint
    if checkpoint_path.exists():
        best_model = build_model(
            model_type=model_type,
            num_classes=config.NUM_CLASSES,
            pretrained=pretrained
        ).to(device)
        best_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        test_loss, test_acc = evaluate(best_model, test_loader, criterion, device)
        history["test_loss"] = test_loss
        history["test_accuracy"] = test_acc
        print(f"Test      - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

    # 7. Save History to JSON
    results_dir = Path("/kaggle/working/results") if getattr(config, 'KAGGLE', False) else Path("./results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    history_path = results_dir / f"history_{model_type}_{model_suffix}_{timestamp}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
        
    print(f"Training completed for {model_type}!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f} (Epoch {history['best_epoch']})")
    print(f"History saved to: {history_path}")

    return history


def main():
    """Main execution block."""
    # Reproducibility
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)

    print(f"Device: {config.DEVICE}")
    print(f"Dataset directory: {config.DATA_DIR}")

    # You can loop through models here for comparisons, or just run the one specified in config
    train_model(
        model_type=config.MODEL_TYPE, 
        pretrained=getattr(config, 'PRETRAINED', True), 
        device=config.DEVICE
    )


if __name__ == "__main__":
    main()