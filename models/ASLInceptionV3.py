import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights

class ASLInceptionV3(nn.Module):
    """
    InceptionV3 model for ASL Classification.
    
    This class supports both pretrained (feature extraction) and from-scratch 
    training modes. 
    
    Note on Input Size: The project specifies 200x200 inputs without scaling. 
    Standard InceptionV3 uses an auxiliary classifier that requires 299x299 inputs. 
    To accommodate the 200x200 constraint, aux_logits is strictly set to False.
    
    Args:
        num_classes (int): Number of output classes (default 26 for letters).
                           Update to 29 for later fine-tuning.
        pretrained (bool): If True, uses ImageNet weights and freezes feature layers.
                           If False, initializes from scratch.
    """
    def __init__(self, num_classes=26, pretrained=True):
        super(ASLInceptionV3, self).__init__()
        
        # Load weights if pretrained, otherwise initialize from scratch
        weights = Inception_V3_Weights.DEFAULT if pretrained else None
        
        # Initialize model. aux_logits=False is REQUIRED for 200x200 inputs.
        self.model = inception_v3(weights=weights, aux_logits=False)
        
        # Feature Extraction Strategy: Freeze all layers initially
        if pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
                
        # Replace the final fully connected layer to match our num_classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
        # Ensure the newly created fc layer is trainable
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Forward pass for the InceptionV3 model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, 200, 200).
            
        Returns:
            torch.Tensor: Output logits of shape (B, num_classes).
        """
        return self.model(x)