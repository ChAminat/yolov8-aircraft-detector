import fire
import numpy as np
import onnxruntime
import torch
from pathlib import Path

from pl_modules.lightning_module import SignLanguageCNNModule


def main(
    checkpoint_path: str,
    onnx_output_path: str = "./models/model.onnx",
    img_size: int = 224,
    batch_size: int = 1
) -> None:
    """
    Convert PyTorch Lightning checkpoint to ONNX format and validate.
    
    Args:
        checkpoint_path: Path to model checkpoint (.ckpt)
        onnx_output_path: Path where to save ONNX model
        img_size: Input image size (default: 224)
        batch_size: Batch size for ONNX export (default: 1)
    """
    # Load model from checkpoint
    module = SignLanguageCNNModule.load_from_checkpoint(checkpoint_path)
    print("✅ Successfully loaded from checkpoint")
    module.eval()
    
    # Create output directory if it doesn't exist
    Path(onnx_output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create dummy input (batch_size, channels, height, width)
    dummy_input = torch.randn(batch_size, 1, img_size, img_size)
    
    # Export to ONNX
    module.to_onnx(
        onnx_output_path,
        dummy_input,
        export_params=True,
        input_names=["INPUT_IMAGE"],
        output_names=["CLASS_LOGITS"],
        dynamic_axes={
            "INPUT_IMAGE": {0: "batch_size"},
            "CLASS_LOGITS": {0: "batch_size"},
        },
        opset_version=11
    )
    
    print(f"✅ Model compiled to ONNX: {onnx_output_path}")
    
    # Validate ONNX model
    ort_session = onnxruntime.InferenceSession(
        onnx_output_path, 
        providers=["CPUExecutionProvider"]
    )
    
    # Get input name
    input_name = ort_session.get_inputs()[0].name
    
    # Run inference
    ort_inputs = {input_name: dummy_input.numpy().astype(np.float32)}
    ort_outs = ort_session.run(None, ort_inputs)
    
    if ort_outs:
        print("✅ ONNX model check passed")
        print(f"   Output shape: {ort_outs[0].shape}")
        print(f"   Sample output: {ort_outs[0][0][:5]} ...")  # first 5 logits
    else:
        print("❌ ONNX model check failed")


if __name__ == "__main__":
    fire.Fire(main)