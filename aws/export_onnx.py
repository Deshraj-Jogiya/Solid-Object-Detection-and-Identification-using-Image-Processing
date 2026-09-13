"""Exports the trained ShapeDetectorNet checkpoint to ONNX so it can run
inside an AWS Lambda zip deployment via onnxruntime, instead of needing the
full PyTorch runtime packaged as a >250MB container image.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import ShapeDetectorNet
from src.dataset import SHAPE_CLASSES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "shape_detector.pth")
ONNX_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "shape_detector.onnx")


def export():
    model = ShapeDetectorNet(num_classes=len(SHAPE_CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, 64, 64)
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["image"],
        output_names=["class_logits", "bbox"],
        dynamic_axes={"image": {0: "batch"}, "class_logits": {0: "batch"}, "bbox": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    size_kb = os.path.getsize(ONNX_PATH) / 1024
    print(f"Exported ONNX model to {ONNX_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    export()
