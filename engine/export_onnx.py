import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.ASLMobileNetV2 import ASLMobileNetV2

def export_onnx():
    checkpoint_path = PROJECT_ROOT / "models" / "checkpoints" / "best_mobilenet_v2_finetuned_29.pth"
    output_dir = PROJECT_ROOT / "docs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_onnx_path = output_dir / "model.onnx"

    print(f"Loading checkpoint: {checkpoint_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Build model (29 classes)
    model = ASLMobileNetV2(num_classes=29, pretrained=False)

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    # Strip 'module.' prefix if present from DP/DDP training
    stripped_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            stripped_state_dict[k[7:]] = v
        else:
            stripped_state_dict[k] = v

    model.load_state_dict(stripped_state_dict, strict=True)
    model.eval()
    print("Model weights loaded successfully.")

    # Create dummy input: 1 batch, 3 channels, 200x200 pixels
    dummy_input = torch.randn(1, 3, 200, 200)

    print(f"Exporting model to ONNX: {output_onnx_path}")
    
    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        str(output_onnx_path),
        export_params=True,
        opset_version=14,  # Opsets 14+ have excellent support in ONNX Runtime Web
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    print("ONNX model exported successfully!")

if __name__ == "__main__":
    export_onnx()
