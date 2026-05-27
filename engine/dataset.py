"""Data loading and preprocessing for ASL Fingerspelling Recognition (PyTorch)."""

import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms

try:
    from . import config
except ImportError:
    import config


class ASLDatasetWrapper(Dataset):
    """
    Wrapper to apply different transforms to subsets. 
    Using random_split creates Subset objects that share the same underlying dataset.
    This wrapper prevents validation data from accidentally receiving training augmentations.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def get_data_loaders(data_dir, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS):
    """
    Load and preprocess ASL dataset with train/validation split.

    Args:
        data_dir: Path to dataset directory containing class folders (A, B, C, ..., Y).
        batch_size: Batch size for DataLoader.
        num_workers: Number of worker processes for data loading.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset)
    """

    # Note: Images are natively 200x200. No resizing or cropping is applied 
    # to maintain input size parity with the dataset constraints.

    # Training transforms: Safe augmentations + preprocessing
    train_transforms = transforms.Compose([
        transforms.RandomAffine(
            degrees=config.ROTATION_DEGREES, 
            translate=(config.TRANSLATION_FRACTION, config.TRANSLATION_FRACTION)
        ),
        transforms.ColorJitter(
            brightness=config.JITTER_BRIGHTNESS,
            contrast=config.JITTER_CONTRAST
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.IMAGENET_MEAN,
            std=config.IMAGENET_STD
        )
    ])

    # Validation transforms: Preprocessing ONLY (no augmentations)
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.IMAGENET_MEAN,
            std=config.IMAGENET_STD
        )
    ])

    # Load full dataset. We apply transforms later via the wrapper.
    # Note: PyTorch ImageFolder uses PIL which loads images directly without transforms initially.
    full_dataset = ImageFolder(root=str(data_dir))

    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(total_size * config.TRAIN_SPLIT)
    val_size = int(total_size * config.VALIDATION_SPLIT)
    test_size = total_size - train_size - val_size

    # Split dataset into train, validation, and test
    train_subset, val_subset, test_subset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )

    # Wrap subsets to safely apply distinct transforms
    train_dataset = ASLDatasetWrapper(train_subset, transform=train_transforms)
    val_dataset = ASLDatasetWrapper(val_subset, transform=val_transforms)
    test_dataset = ASLDatasetWrapper(test_subset, transform=val_transforms)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Mixing everything up to prevent background/lighting clustering
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"{'-'*50}")
    print(f"Dataset loaded successfully!")
    print(f"Total samples: {total_size}")
    print(f"Training samples: {train_size} ({config.TRAIN_SPLIT * 100:.1f}%)")
    print(f"Validation samples: {val_size} ({config.VALIDATION_SPLIT * 100:.1f}%)")
    print(f"Test samples: {test_size} ({config.TEST_SPLIT * 100:.1f}%)")
    print(f"Number of classes: {len(full_dataset.classes)}")
    print(f"{'-'*50}")
    
    # Mandatory report justification for missing subject IDs
    print("\n SPLITTING STRATEGY LIMITATION WARNING ")
    print("Due to absence of subject labels, a strict subject-independent split is not possible.")
    print("This random split may lead to optimistic performance estimates as the same person's "
          "hand may appear in both training and test sets.")
    print(f"{'-'*50}\n")

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset