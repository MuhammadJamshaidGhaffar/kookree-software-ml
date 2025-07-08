
"""
Export ResNet-18 from torchvision to ONNX format (opset 11, NCHW 224×224).

Usage:
    python export_to_onnx.py

This script should be run once inside the Docker build context to generate
the ONNX model file for inference. The output file will be saved as:
    inference_service/model/resnet18.onnx
"""
import torch
from torchvision import models
import os

PATH = os.path.join("inference_service", "model", "resnet18.onnx")



def export():
    """
    Export torchvision's ResNet-18 to ONNX format for inference.
    The ONNX file will be saved to PATH.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, PATH,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
    )
    print(f"Exported ResNet‑18 → {PATH}")
    print(f"File is saved in: {os.path.abspath(PATH)}")


if __name__ == "__main__":
    export()
