import torch.nn as nn

class ASLCustomCNN(nn.Module):
    """
    Custom Convolutional Neural Network for ASL fingerspelling classification.

    Architecture:
    - 4 convolutional blocks (Conv -> BatchNorm -> ReLU -> MaxPool)
    - Adaptive average pooling to flatten spatial dimensions
    - Classifier head with dropout and linear layer

    Input: (B, 3, 224, 224) RGB images
    Output: (B, num_classes) classification logits
    """

    def __init__(self, num_classes=24):
        """
        Initialize the ASL Custom CNN.

        Args:
            num_classes: Number of output classes.
        """
        super(ASLCustomCNN, self).__init__()

        # Convolutional Block 1: 3 -> 32 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Convolutional Block 2: 32 -> 64 channels
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Convolutional Block 3: 64 -> 128 channels
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Convolutional Block 4: 128 -> 256 channels
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Adaptive average pooling: reduces spatial dimensions to 1x1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head: Dropout + Linear layer
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

        # Initialize weights using Kaiming He initialization
        self._initialize_weights()

        # Print model summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"ASL Custom CNN initialized!")
        print(f"Number of classes: {num_classes}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def _initialize_weights(self):
        """Initialize weights using Kaiming He initialization for Conv2d layers."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Convolutional feature extraction blocks
        x = self.block1(x)      # (B, 32, H/2, W/2)
        x = self.block2(x)      # (B, 64, H/4, W/4)
        x = self.block3(x)      # (B, 128, H/8, W/8)
        x = self.block4(x)      # (B, 256, H/16, W/16)

        # Adaptive average pooling to 1x1
        x = self.adaptive_pool(x)  # (B, 256, 1, 1)

        # Flatten spatial dimensions
        x = x.view(x.size(0), -1)  # (B, 256)

        # Classification head
        x = self.classifier(x)  # (B, num_classes)

        return x