"""Real, one-time (re-runnable/idempotent) deploy script: creates the S3
bucket, packages the Lambda zip (onnxruntime + numpy + Pillow as
manylinux2014_x86_64 wheels -- no container image, so this never touches the
ECR storage that sits outside the AWS Free Tier), creates/updates the Lambda
function, and wires up the S3 -> Lambda event trigger on the `incoming/`
prefix.

Needs AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION in the
environment (the `career-pilot-integration` IAM user's key) and the
`career-pilot-shape-detector-lambda-role` execution role already created.

Run: python aws/deploy.py
"""
import io
import json
import os
import subprocess
import sys
import zipfile

import boto3
from botocore.exceptions import ClientError

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

ACCOUNT_ID = "964862484434"
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
BUCKET_NAME = f"career-pilot-shape-detector-{ACCOUNT_ID}"
FUNCTION_NAME = "career-pilot-shape-detector"
ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/career-pilot-shape-detector-lambda-role"
LAMBDA_RUNTIME = "python3.12"
PACKAGE_DIR = os.path.join(HERE, "_package")
ZIP_PATH = os.path.join(HERE, "_deployment.zip")

s3 = boto3.client("s3", region_name=REGION)
lam = boto3.client("lambda", region_name=REGION)


def ensure_bucket():
    try:
        s3.head_bucket(Bucket=BUCKET_NAME)
        print(f"Bucket {BUCKET_NAME} already exists")
    except ClientError:
        if REGION == "us-east-1":
            s3.create_bucket(Bucket=BUCKET_NAME)
        else:
            s3.create_bucket(
                Bucket=BUCKET_NAME,
                CreateBucketConfiguration={"LocationConstraint": REGION},
            )
        print(f"Created bucket {BUCKET_NAME}")

    s3.put_public_access_block(
        Bucket=BUCKET_NAME,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )

    # Lifecycle rule: auto-delete everything after 3 days. This is a free
    # personal-project demo bucket, not a real data store -- capping how
    # long objects stick around keeps storage trivially inside the Free
    # Tier's 5GB even if left running unattended.
    s3.put_bucket_lifecycle_configuration(
        Bucket=BUCKET_NAME,
        LifecycleConfiguration={
            "Rules": [
                {
                    "ID": "expire-after-3-days",
                    "Status": "Enabled",
                    "Filter": {"Prefix": ""},
                    "Expiration": {"Days": 3},
                }
            ]
        },
    )


def build_package():
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "--target", PACKAGE_DIR,
            "--platform", "manylinux_2_28_x86_64",
            "--implementation", "cp",
            "--python-version", "3.12",
            "--only-binary=:all:",
            "--upgrade",
            "onnxruntime==1.30.0", "numpy==2.5.3", "pillow==12.3.0",
        ],
        check=True,
    )

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(PACKAGE_DIR):
            for f in files:
                full = os.path.join(root, f)
                arcname = os.path.relpath(full, PACKAGE_DIR)
                zf.write(full, arcname)

        zf.write(os.path.join(HERE, "lambda_function.py"), "lambda_function.py")
        zf.write(os.path.join(REPO_ROOT, "models", "shape_detector.onnx"), "shape_detector.onnx")

    size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    print(f"Built deployment zip: {ZIP_PATH} ({size_mb:.1f} MB)")
    return size_mb


def ensure_function():
    # The zip (~53MB, dominated by onnxruntime) is over Lambda's ~50MB
    # direct-upload (inline ZipFile) limit, so stage it through S3 instead --
    # CreateFunction/UpdateFunctionCode support up to 250MB unzipped that way.
    deploy_key = "_deploy/lambda_function.zip"
    s3.upload_file(ZIP_PATH, BUCKET_NAME, deploy_key)
    print(f"Uploaded deployment zip to s3://{BUCKET_NAME}/{deploy_key}")

    try:
        lam.get_function(FunctionName=FUNCTION_NAME)
        print(f"Updating existing function {FUNCTION_NAME}")
        lam.update_function_code(FunctionName=FUNCTION_NAME, S3Bucket=BUCKET_NAME, S3Key=deploy_key)
        lam.get_waiter("function_updated").wait(FunctionName=FUNCTION_NAME)
        lam.update_function_configuration(
            FunctionName=FUNCTION_NAME,
            Runtime=LAMBDA_RUNTIME,
            Handler="lambda_function.handler",
            Role=ROLE_ARN,
            Timeout=30,
            MemorySize=512,
        )
    except lam.exceptions.ResourceNotFoundException:
        print(f"Creating function {FUNCTION_NAME}")
        lam.create_function(
            FunctionName=FUNCTION_NAME,
            Runtime=LAMBDA_RUNTIME,
            Role=ROLE_ARN,
            Handler="lambda_function.handler",
            Code={"S3Bucket": BUCKET_NAME, "S3Key": deploy_key},
            Timeout=30,
            MemorySize=512,
            Description="Real-time shape classification triggered by S3 uploads (career-pilot skill-gap project)",
        )
        lam.get_waiter("function_active").wait(FunctionName=FUNCTION_NAME)

    return lam.get_function(FunctionName=FUNCTION_NAME)["Configuration"]["FunctionArn"]


def ensure_s3_trigger(function_arn):
    statement_id = "AllowS3Invoke"
    try:
        lam.add_permission(
            FunctionName=FUNCTION_NAME,
            StatementId=statement_id,
            Action="lambda:InvokeFunction",
            Principal="s3.amazonaws.com",
            SourceArn=f"arn:aws:s3:::{BUCKET_NAME}",
            SourceAccount=ACCOUNT_ID,
        )
        print("Added S3 invoke permission")
    except lam.exceptions.ResourceConflictException:
        print("S3 invoke permission already present")

    s3.put_bucket_notification_configuration(
        Bucket=BUCKET_NAME,
        NotificationConfiguration={
            "LambdaFunctionConfigurations": [
                {
                    "LambdaFunctionArn": function_arn,
                    "Events": ["s3:ObjectCreated:*"],
                    "Filter": {
                        "Key": {
                            "FilterRules": [{"Name": "prefix", "Value": "incoming/"}]
                        }
                    },
                }
            ]
        },
    )
    print("Wired S3 ObjectCreated (incoming/*) -> Lambda trigger")


def main():
    ensure_bucket()
    build_package()
    function_arn = ensure_function()
    ensure_s3_trigger(function_arn)
    print(json.dumps({"bucket": BUCKET_NAME, "function_arn": function_arn}, indent=2))


if __name__ == "__main__":
    main()
