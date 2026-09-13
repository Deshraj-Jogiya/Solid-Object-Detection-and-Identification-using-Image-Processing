"""Real AWS Lambda handler: triggered by an S3 ObjectCreated event under the
`incoming/` prefix, runs the actual trained ShapeDetectorNet (exported to
ONNX, see export_onnx.py) on the uploaded image via onnxruntime, and writes
the classification + bounding box result back to the same bucket under
`results/<original-key>.json`.

Kept dependency-light on purpose (onnxruntime + numpy + Pillow, no OpenCV/
PyTorch/boto3-heavy deps) so this ships as a plain Lambda zip well under the
250MB unzipped limit -- no container image, so nothing here can run into the
Free Tier's ECR storage allowance.
"""
import io
import json
import os

import boto3
import numpy as np
import onnxruntime as ort
from PIL import Image

SHAPE_CLASSES = ["circle", "square", "triangle", "rectangle", "pentagon", "hexagon", "star"]

_HERE = os.path.dirname(__file__)
# In the deployed Lambda zip the model sits flat next to this file; in the
# repo checkout (local dev/tests) it lives in ../models/ alongside the .pth
# checkpoint, same convention as the rest of the repo.
_MODEL_PATH = os.path.join(_HERE, "shape_detector.onnx")
if not os.path.exists(_MODEL_PATH):
    _MODEL_PATH = os.path.join(_HERE, "..", "models", "shape_detector.onnx")
_session = None

s3 = boto3.client("s3")


def _get_session():
    global _session
    if _session is None:
        _session = ort.InferenceSession(_MODEL_PATH, providers=["CPUExecutionProvider"])
    return _session


def _preprocess(image_bytes: bytes) -> tuple[np.ndarray, int, int]:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    resized = img.resize((64, 64))
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr, w, h


def predict(image_bytes: bytes) -> dict:
    tensor, w, h = _preprocess(image_bytes)
    session = _get_session()
    logits, bbox = session.run(None, {"image": tensor})

    logits = logits[0]
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    class_idx = int(np.argmax(probs))
    confidence = float(probs[class_idx])

    xmin, ymin, xmax, ymax = bbox[0]
    result_bbox = [
        max(0, min(int(xmin * w), w - 1)),
        max(0, min(int(ymin * h), h - 1)),
        max(0, min(int(xmax * w), w - 1)),
        max(0, min(int(ymax * h), h - 1)),
    ]

    return {
        "label_name": SHAPE_CLASSES[class_idx],
        "confidence": confidence,
        "bbox": result_bbox,
    }


def handler(event, context):
    results = []
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]

        obj = s3.get_object(Bucket=bucket, Key=key)
        image_bytes = obj["Body"].read()

        result = predict(image_bytes)

        result_key = key.replace("incoming/", "results/", 1) + ".json"
        s3.put_object(
            Bucket=bucket,
            Key=result_key,
            Body=json.dumps(result).encode("utf-8"),
            ContentType="application/json",
        )
        results.append({"source_key": key, "result_key": result_key, "result": result})

    return {"statusCode": 200, "body": json.dumps(results)}
