"""Data loading and preprocessing for ASL Fingerspelling Recognition (PyTorch) - Custom CNN."""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms

import config


def get_data_loaders(data_dir, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS):
    """
    Load and preprocess ASL dataset with train/validation split.

    For a custom CNN trained from scratch, strong data augmentation is critical
    to prevent overfitting and improve generalization.

    Args:
        data_dir: Path to dataset directory containing class folders (A, B, C, ..., Y).
        batch_size: Batch size for DataLoader.
        num_workers: Number of worker processes for data loading.

    Returns:
        Tuple of (train_loader, val_loader, train_dataset, val_dataset)
    """

    # Training transforms: augmentation + preprocessing
    train_transforms = transforms.Compose([
        transforms.Resize(256),  # Resize to 256x256
        transforms.RandomCrop(config.IMAGE_SIZE),  # Random crop to 224x224
        transforms.RandomHorizontalFlip(p=config.HORIZONTAL_FLIP_PROB),  # Horizontal flip with 50% probability
        transforms.RandomRotation(config.ROTATION_DEGREES),  # Random rotation ±20 degrees
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(
            mean=config.IMAGENET_MEAN,
            std=config.IMAGENET_STD
        )
    ])

    # Validation transforms: no augmentation, just preprocessing
    val_transforms = transforms.Compose([
        transforms.Resize(256),  # Resize to 256x256
        transforms.CenterCrop(config.IMAGE_SIZE),  # Center crop to 224x224
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(
            mean=config.IMAGENET_MEAN,
            std=config.IMAGENET_STD
        )
    ])

    # Load full dataset with no specific transforms yet
    full_dataset = ImageFolder(root=str(data_dir))

    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(total_size * config.TRAIN_SPLIT)
    val_size = total_size - train_size

    # Split dataset into train and validation
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )

    # Apply transforms to train and validation subsets
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Dataset loaded successfully!")
    print(f"Total samples: {total_size}")
    print(f"Training samples: {train_size} ({config.TRAIN_SPLIT * 100:.1f}%)")
    print(f"Validation samples: {val_size} ({config.VALIDATION_SPLIT * 100:.1f}%)")
    print(f"Number of classes: {len(full_dataset.classes)}")
    print(f"Classes: {full_dataset.classes}")

    return train_loader, val_loader, train_dataset, val_dataset
