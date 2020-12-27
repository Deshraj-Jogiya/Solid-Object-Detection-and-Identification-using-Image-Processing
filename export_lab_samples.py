"""Exports held-out shapes for the portfolio Lab demo, with the ONNX model's outputs on each one.

Takes the first 9 images of each class from the same seed-42 validation split verify_accuracy.py scores, saves them as PNG
(lossless, so the browser reads exactly the pixels the model saw) and writes samples.json with each image's true label and
the model's probabilities. The Lab runs the same ONNX file in the browser and must match these probabilities (parity check).

    python export_lab_samples.py <output_dir>
"""
import json
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import random_split

from src.dataset import SHAPE_CLASSES, ShapeDataset

PER_CLASS = 9
MODEL = os.path.join("models", "shape_detector.onnx")


def main(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    full = ShapeDataset(os.path.join("data", "shapes"))
    train_size = int(0.8 * len(full))
    _, val = random_split(full, [train_size, len(full) - train_size], generator=torch.Generator().manual_seed(42))

    session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    taken = {name: 0 for name in SHAPE_CLASSES}
    samples = []
    for idx in val.indices:
        item = full.annotations[idx]
        if taken[item["label_name"]] >= PER_CLASS:
            continue
        rgb = cv2.cvtColor(cv2.imread(os.path.join(full.images_dir, item["filename"])), cv2.COLOR_BGR2RGB)
        tensor = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None]
        logits = session.run(None, {input_name: tensor})[0][0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        name = item["filename"].replace(".jpg", ".png")
        cv2.imwrite(os.path.join(out_dir, name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        samples.append({"file": name, "label": item["label_name"],
                        "probs": [round(float(p), 6) for p in probs],
                        "predicted": SHAPE_CLASSES[int(np.argmax(probs))]})
        taken[item["label_name"]] += 1
        if all(v >= PER_CLASS for v in taken.values()):
            break

    correct = sum(s["predicted"] == s["label"] for s in samples)
    with open(os.path.join(out_dir, "samples.json"), "w") as f:
        json.dump({"classes": SHAPE_CLASSES, "input": input_name, "size": 64, "samples": samples}, f, indent=1)
    print(f"{len(samples)} held-out images exported, model correct on {correct}")


if __name__ == "__main__":
    main(sys.argv[1])
