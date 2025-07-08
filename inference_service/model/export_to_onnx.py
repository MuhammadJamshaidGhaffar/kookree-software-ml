"""
Run once **inside the Docker build** to export ResNet‑18
from torchvision to ONNX (opset 11, NCHW 224×224).
"""
import torch
from torchvision import models
import os

PATH = os.path.join("inference_service", "model", "resnet18.onnx")


def export():
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
    # print current direclty where this fiile will be saved
    print(f"File is saved in: {os.path.abspath(PATH)}")

if __name__ == "__main__":
    export()
