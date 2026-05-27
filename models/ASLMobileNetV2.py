import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class ASLMobileNetV2(nn.Module):
    """
    MobileNetV2 model for ASL Classification.
    
    This class supports two modes for the project's comparison design:
    1. Pretrained (Feature Extraction): Freezes all convolutional features and 
       only trains the final classification head.
    2. From Scratch: Initializes the architecture with random weights for full training.
    
    Args:
        num_classes (int): Number of output classes (default 26 for letters). 
                           Update to 29 for later fine-tuning (space, delete, nothing).
        pretrained (bool): If True, uses ImageNet weights and freezes feature layers. 
                           If False, initializes from scratch.
    """
    def __init__(self, num_classes=26, pretrained=True):
        super(ASLMobileNetV2, self).__init__()
        
        # Load weights if pretrained, otherwise initialize from scratch
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.model = mobilenet_v2(weights=weights)
        
        # Feature Extraction Strategy: Freeze early layers
        if pretrained:
            for param in self.model.features.parameters():
                param.requires_grad = False
                
        # Replace the classifier head to match our number of classes
        # The new layer automatically has requires_grad=True
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass for the MobileNetV2 model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, 200, 200).
            
        Returns:
            torch.Tensor: Output logits of shape (B, num_classes).
        """
        return self.model(x)