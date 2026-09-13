"""Real tests for the Lambda handler's inference logic (no mocking of the
ONNX runtime or the model -- runs the actual exported model). S3-trigger
wiring itself is exercised end-to-end by tests/test_s3_integration.py against
a real live bucket.
"""
import io
import os
import sys
import unittest

import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lambda_function import predict
from src.dataset import generate_shape_image, SHAPE_CLASSES


def _shape_to_png_bytes(shape_type: str) -> bytes:
    img, _ = generate_shape_image(shape_type, img_size=128)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[2] == 3 else img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


class TestLambdaPredict(unittest.TestCase):
    def test_predicts_correct_shape_for_each_class(self):
        # Real end-to-end: generate a real synthetic shape image the same way
        # the training data was generated, run it through the real exported
        # ONNX model via the actual Lambda predict() function, and check the
        # label matches -- same real-model verification style as
        # verify_accuracy.py, just via the Lambda code path.
        for shape in SHAPE_CLASSES:
            with self.subTest(shape=shape):
                image_bytes = _shape_to_png_bytes(shape)
                result = predict(image_bytes)
                self.assertEqual(result["label_name"], shape)
                self.assertGreater(result["confidence"], 0.5)
                self.assertEqual(len(result["bbox"]), 4)

    def test_bbox_is_within_image_bounds(self):
        image_bytes = _shape_to_png_bytes("circle")
        result = predict(image_bytes)
        xmin, ymin, xmax, ymax = result["bbox"]
        self.assertTrue(0 <= xmin < xmax <= 128)
        self.assertTrue(0 <= ymin < ymax <= 128)


if __name__ == "__main__":
    unittest.main()
