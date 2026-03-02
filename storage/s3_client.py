"""
Async S3-compatible storage client.
Supports AWS S3 and MinIO via endpoint_url override.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import aioboto3
from botocore.config import Config
from botocore.exceptions import ClientError

from config import settings

logger = logging.getLogger(__name__)


class S3Client:
    """Thin async wrapper around aioboto3 for S3-compatible object storage."""

    def __init__(self) -> None:
        self._session = aioboto3.Session(
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY,
            region_name=settings.S3_REGION,
        )
        self._endpoint_url: Optional[str] = settings.S3_ENDPOINT_URL or None
        self._bucket = settings.S3_BUCKET_NAME
        self._config = Config(retries={"max_attempts": 5, "mode": "standard"})

    def _client_kwargs(self) -> dict:
        kwargs: dict = {"config": self._config}
        if self._endpoint_url:
            kwargs["endpoint_url"] = self._endpoint_url
        return kwargs

    async def ensure_bucket(self) -> None:
        """Create the bucket if it does not exist (useful for MinIO)."""
        async with self._session.client("s3", **self._client_kwargs()) as s3:
            try:
                await s3.head_bucket(Bucket=self._bucket)
                logger.debug("Bucket %s already exists", self._bucket)
            except ClientError as e:
                if e.response["Error"]["Code"] in ("404", "NoSuchBucket"):
                    await s3.create_bucket(Bucket=self._bucket)
                    logger.info("Created bucket %s", self._bucket)
                else:
                    raise

    async def upload_file(
        self,
        local_path: str,
        object_key: str,
        content_type: str = "video/mp4",
        extra_args: Optional[dict] = None,
    ) -> str:
        """Upload a local file and return its object key."""
        upload_args = {"ContentType": content_type}
        if extra_args:
            upload_args.update(extra_args)

        logger.info("Uploading %s → s3://%s/%s", local_path, self._bucket, object_key)
        async with self._session.client("s3", **self._client_kwargs()) as s3:
            await s3.upload_file(
                local_path,
                self._bucket,
                object_key,
                ExtraArgs=upload_args,
            )
        logger.info("Upload complete: %s", object_key)
        return object_key

    async def generate_presigned_url(
        self,
        object_key: str,
        expiry: int = 3600,
    ) -> str:
        """Return a presigned GET URL valid for `expiry` seconds."""
        async with self._session.client("s3", **self._client_kwargs()) as s3:
            url: str = await s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": object_key},
                ExpiresIn=expiry,
            )
        return url

    async def upload_and_sign(
        self,
        local_path: str,
        object_key: str,
        expiry: int = 3600,
        content_type: str = "video/mp4",
    ) -> str:
        """Upload file and return a presigned download URL."""
        await self.upload_file(local_path, object_key, content_type)
        return await self.generate_presigned_url(object_key, expiry)


# Module-level singleton
s3_client = S3Client()
