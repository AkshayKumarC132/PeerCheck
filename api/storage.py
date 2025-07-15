import boto3
import hashlib
import os
from botocore.exceptions import ClientError
from uuid import uuid4
from typing import Dict

class S3Storage:
    def __init__(self, bucket: str, region: str = "us-east-1"):
        self.bucket = bucket
        self.s3 = boto3.client("s3", region_name=region)

    def upload_file(self, fileobj, filename: str, key_prefix: str, metadata: Dict[str, str]) -> str:
        """Upload a file-like object to S3 with metadata tags and integrity verification.

        Parameters
        ----------
        fileobj: file-like object opened in binary mode
            The file object to upload. The caller is responsible for closing it.
        filename: str
            Original filename used for generating the S3 key.
        key_prefix: str
            Prefix to namespace the object within the bucket.
        metadata: Dict[str, str]
            Extra metadata to store with the object.
        Returns
        -------
        str
            The HTTPS URL of the uploaded object.
        """

        file_key = f"{key_prefix}/{uuid4()}-{os.path.basename(filename)}"
        tags = "&".join([f"{k}={v}" for k, v in metadata.items()])
        try:
            data = fileobj.read()
            md5 = hashlib.md5(data).hexdigest()
            self.s3.put_object(
                Bucket=self.bucket,
                Key=file_key,
                Body=data,
                Tagging=tags,
                Metadata={"md5": md5, **metadata},
            )
            head = self.s3.head_object(Bucket=self.bucket, Key=file_key)
            if head.get("Metadata", {}).get("md5") != md5:
                raise RuntimeError("S3 integrity check failed")
        except ClientError as e:
            raise RuntimeError(f"S3 upload failed: {e}")

        url = f"https://{self.bucket}.s3.{self.s3.meta.region_name}.amazonaws.com/{file_key}"
        return url

    def download_file(self, key_or_url: str, dest: str) -> None:
        """Download a file from S3 to a local destination."""
        if key_or_url.startswith("http"):
            # Strip the bucket domain to obtain the key
            key = key_or_url.split(f"{self.bucket}/")[-1]
        else:
            key = key_or_url
        try:
            self.s3.download_file(self.bucket, key, dest)
        except ClientError as e:
            raise RuntimeError(f"S3 download failed: {e}")

    def verify_file(self, file_key: str) -> bool:
        """Check that the file exists in the bucket."""
        try:
            self.s3.head_object(Bucket=self.bucket, Key=file_key)
            return True
        except ClientError:
            return False
