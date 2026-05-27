"""Training comparison script - trains both models and compares results."""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
from datetime import datetime

import config as original_config
from dataset import get_data_loaders
from model import build_model


def train_and_evaluate_model(model_type, train_loader, val_loader, device):
    """
    Train and evaluate a single model.

    Args:
        model_type: "custom_cnn" or "mobilenet_v2"
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        device: Device to train on

    Returns:
        Dictionary with training results
    """
    # Temporarily set model type in config
    original_config.MODEL_TYPE = model_type

    print(f"\n{'='*70}")
    print(f"Training {model_type.upper()}")
    print(f"{'='*70}\n")

    # Build model
    model = build_model(num_classes=original_config.NUM_CLASSES)
    model = model.to(device)
    print()

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Select optimizer based on model type
    if model_type == "custom_cnn":
        optimizer = optim.SGD(
            model.parameters(),
            lr=original_config.LEARNING_RATE,
            momentum=original_config.OPTIMIZER_MOMENTUM,
            weight_decay=original_config.OPTIMIZER_WEIGHT_DECAY
        )
    else:  # mobilenet_v2
        optimizer = optim.Adam(
            model.classifier.parameters(),
            lr=original_config.LEARNING_RATE
        )

    # Training loop
    best_val_accuracy = 0.0
    training_history = {
        "model": model_type,
        "epochs": original_config.EPOCHS,
        "learning_rate": original_config.LEARNING_RATE,
        "batch_size": original_config.BATCH_SIZE,
        "train_losses": [],
        "train_accuracies": [],
        "val_losses": [],
        "val_accuracies": [],
        "best_val_accuracy": 0.0,
        "best_epoch": 0
    }

    print(f"Starting training for {original_config.EPOCHS} epochs...\n")

    for epoch in range(original_config.EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

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
                print(f"  Epoch [{epoch + 1}/{original_config.EPOCHS}], "
                      f"Batch [{batch_idx + 1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_predictions / total_samples

        # Validation phase
        model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        val_loss = running_loss / len(val_loader)
        val_accuracy = correct_predictions / total_samples

        # Store metrics
        training_history["train_losses"].append(train_loss)
        training_history["train_accuracies"].append(train_accuracy)
        training_history["val_losses"].append(val_loss)
        training_history["val_accuracies"].append(val_accuracy)

        # Print epoch statistics
        print(f"Training   - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            training_history["best_val_accuracy"] = best_val_accuracy
            training_history["best_epoch"] = epoch + 1

            checkpoint_dir = Path("/kaggle/working")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"best_model_{model_type}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Best model saved (val_acc: {best_val_accuracy:.4f})\n")
        else:
            print()

    training_history["best_val_accuracy"] = best_val_accuracy

    print(f"Training completed for {model_type}!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")

    return training_history


def main():
    """Main comparison pipeline."""

    # Set random seeds
    torch.manual_seed(original_config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(original_config.SEED)

    print(f"Device: {original_config.DEVICE}")
    print(f"Dataset directory: {original_config.DATA_DIR}\n")

    # Create output directory
    output_dir = Path("/kaggle/working")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data once
    print("Loading dataset...")
    train_loader, val_loader, train_dataset, val_dataset = get_data_loaders(
        data_dir=original_config.DATA_DIR,
        batch_size=original_config.BATCH_SIZE,
        num_workers=original_config.NUM_WORKERS
    )
    print()

    # Train both models
    results = {}

    for model_type in ["custom_cnn", "mobilenet_v2"]:
        results[model_type] = train_and_evaluate_model(
            model_type,
            train_loader,
            val_loader,
            original_config.DEVICE
        )

    # Print comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70 + "\n")

    print(f"{'Model':<20} {'Val Accuracy':<15} {'Best Epoch':<15}")
    print("-" * 50)

    for model_type, history in results.items():
        print(f"{model_type:<20} {history['best_val_accuracy']:.4f}          {history['best_epoch']:<15}")

    # Calculate difference
    custom_cnn_acc = results["custom_cnn"]["best_val_accuracy"]
    mobilenet_acc = results["mobilenet_v2"]["best_val_accuracy"]
    diff = custom_cnn_acc - mobilenet_acc

    print("\n" + "-" * 50)
    if diff > 0:
        print(f"Custom CNN performed better by {abs(diff):.4f} ({abs(diff)*100:.2f}%)")
    else:
        print(f"MobileNetV2 performed better by {abs(diff):.4f} ({abs(diff)*100:.2f}%)")

    # Save results to JSON
    results_path = output_dir / "comparison_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")

    print("\n✓ Training comparison complete!")


if __name__ == "__main__":
    main()
