"""Cross-check the CNN's prediction with the contour-geometry classifier.

The CNN alone is the more accurate method (99.2% on the held-out images, against 66.6% for contour geometry), so it always makes the call.
Contour geometry is used as an independent check: when both methods name the same shape the prediction is marked as confirmed. On the
held-out images the two agree on about two thirds of them, and those confirmed predictions are right 99.9% of the time, so the flag tells a
user which results need no second look. Measured by verify_accuracy.py.
"""

import cv2
import torch

from src.dataset import SHAPE_CLASSES
from src.traditional import detect_shapes_traditional


def geometry_label(image_bgr) -> str | None:
    """Label of the largest shape the contour classifier finds, or None."""
    detections = detect_shapes_traditional(image_bgr)
    if not detections:
        return None
    return max(detections, key=lambda d: cv2.contourArea(d["contour"]))["label_name"]


def classify_with_check(model, image_tensor, image_bgr) -> dict:
    """CNN label and confidence, the geometry label, and whether the two agree."""
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0))[0]
        probs = torch.softmax(logits, 1)[0]
    cnn = SHAPE_CLASSES[int(probs.argmax())]
    geo = geometry_label(image_bgr)
    return {"label": cnn, "confidence": float(probs.max()), "geometry_label": geo, "confirmed": geo == cnn}
