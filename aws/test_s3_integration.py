"""Real, live end-to-end integration test: uploads a real synthetic shape
image to the actual deployed S3 bucket under incoming/, waits for the real
deployed Lambda (triggered by the real S3 ObjectCreated event) to write its
classification result back to results/, and checks it.

No mocking anywhere -- this is the actual AWS infrastructure (S3 event
notification -> Lambda -> S3 write-back) created by aws/deploy.py. Needs
AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION in the
environment and the infrastructure already deployed.
"""
import io
import json
import os
import sys
import time
import unittest

import boto3
import cv2
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dataset import generate_shape_image

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
BUCKET_NAME = "career-pilot-shape-detector-964862484434"
POLL_TIMEOUT_S = 180


def _make_test_image(shape_type: str) -> bytes:
    img, _ = generate_shape_image(shape_type, img_size=128)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


class TestS3LambdaIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s3 = boto3.client("s3", region_name=REGION)

    def _upload_and_wait(self, shape_type: str) -> dict:
        key = f"incoming/ci_test_{shape_type}_{int(time.time())}.png"
        result_key = key.replace("incoming/", "results/", 1) + ".json"

        self.s3.put_object(
            Bucket=BUCKET_NAME, Key=key, Body=_make_test_image(shape_type), ContentType="image/png"
        )

        deadline = time.time() + POLL_TIMEOUT_S
        while time.time() < deadline:
            try:
                obj = self.s3.get_object(Bucket=BUCKET_NAME, Key=result_key)
                return json.loads(obj["Body"].read().decode("utf-8"))
            except self.s3.exceptions.NoSuchKey:
                time.sleep(3)
        self.fail(f"No result written to s3://{BUCKET_NAME}/{result_key} within {POLL_TIMEOUT_S}s")

    def test_real_upload_triggers_real_lambda_classification(self):
        result = self._upload_and_wait("triangle")
        self.assertEqual(result["label_name"], "triangle")
        self.assertGreater(result["confidence"], 0.5)
        self.assertEqual(len(result["bbox"]), 4)


if __name__ == "__main__":
    unittest.main()
