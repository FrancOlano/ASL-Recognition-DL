try:
    from . import config
except ImportError:
    import config

# Importing models from the /models directory
from models.ASLCustomCNN import ASLCustomCNN
from models.ASLMobileNetV2 import ASLMobileNetV2
from models.ASLInceptionV3 import ASLInceptionV3


def _print_model_summary(model_name, model, num_classes, pretrained):
    """
    Generalized helper function to print the parameter counts and training strategy.
    
    Args:
        model_name (str): The display name of the model.
        model (nn.Module): The instantiated PyTorch model.
        num_classes (int): Number of output classes.
        pretrained (bool): Whether the model is using transfer learning.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'-'*50}")
    print(f"{model_name} built successfully!")
    print(f"Number of classes: {num_classes}")
    print(f"Total parameters: {total_params:,}")
    
    # Custom CNN is always from scratch, so we handle its string formatting explicitly
    if pretrained and model_name != "Custom CNN":
        print(f"Trainable parameters: {trainable_params:,} (classifier only)")
        print(f"Training strategy: Feature extraction (base model frozen)")
    else:
        print(f"Trainable parameters: {trainable_params:,} (all layers)")
        print(f"Training strategy: Training from scratch")
    print(f"{'-'*50}\n")


def build_model(model_type=config.MODEL_TYPE, num_classes=config.NUM_CLASSES, pretrained=True):
    """
    Factory function to build the appropriate model based on the requested type.

    Supported models:
    - "custom_cnn": Custom CNN trained from scratch.
    - "mobilenet_v2": MobileNetV2 (supports pretrained feature extraction or scratch).
    - "inception_v3": InceptionV3 (supports pretrained feature extraction or scratch).

    Args:
        model_type (str): Identifier for the model architecture.
        num_classes (int): Number of output classes (26 for A-Z initially).
        pretrained (bool): If True, uses transfer learning (freezes base layers). 
                           If False, initializes the model from scratch.

    Returns:
        Instantiated model ready for training.
    """
    print(f"\n{'='*70}")
    print(f"Building model: {model_type.upper()}")
    
    # Print intent (Custom CNN ignores the pretrained flag conceptually, but we clarify it here)
    mode_str = "Pretrained (Feature Extraction)" if pretrained and model_type != "custom_cnn" else "From Scratch"
    print(f"Requested Mode: {mode_str}")
    print(f"{'='*70}")

    if model_type == "custom_cnn":
        # Custom CNN [cite: 181] does not use pretrained weights; it is always trained from scratch [cite: 81]
        model = ASLCustomCNN(num_classes=num_classes)
        _print_model_summary("Custom CNN", model, num_classes, pretrained=False)
        
    elif model_type == "mobilenet_v2":
        # MobileNetV2 from scratch [cite: 183] or pretrained [cite: 182]
        model = ASLMobileNetV2(num_classes=num_classes, pretrained=pretrained)
        actual_pretrained = getattr(model, "pretrained", pretrained)
        _print_model_summary("MobileNetV2", model, num_classes, actual_pretrained)
        
    elif model_type == "inception_v3":
        # InceptionV3 from scratch [cite: 185] or pretrained [cite: 184]
        model = ASLInceptionV3(num_classes=num_classes, pretrained=pretrained)
        actual_pretrained = getattr(model, "pretrained", pretrained)
        _print_model_summary("InceptionV3", model, num_classes, actual_pretrained)
        
    else:
        raise ValueError(
            f"Unknown MODEL_TYPE: '{model_type}'. "
            "Choose 'custom_cnn', 'mobilenet_v2', or 'inception_v3'."
        )

    return model