# src/ingestion/users/writer.py

import json
from pathlib import Path
from typing import List, Dict

from google.cloud import storage


class UserWriter:
    def __init__(
        self,
        mode: str = "local",
        base_path: str = "data/bronze/users",
        bucket_name: str | None = None,
        gcs_prefix: str = "users/synthetic"
    ):
        self.mode = mode

        if mode == "local":
            self.base_path = Path(base_path)
            self.base_path.mkdir(parents=True, exist_ok=True)

        elif mode == "gcs":
            if not bucket_name:
                raise ValueError("bucket_name required for GCS mode")

            self.client = storage.Client()
            self.bucket = self.client.bucket(bucket_name)
            self.gcs_prefix = gcs_prefix

        else:
            raise ValueError(f"Unsupported writer mode: {mode}")

    def write(self, users: List[Dict], filename: str) -> None:
        """
        Persist raw user snapshots.
        """
        if self.mode == "local":
            self._write_local(users, filename)
        elif self.mode == "gcs":
            self._write_gcs(users, filename)

    def _write_local(self, users: List[Dict], filename: str) -> None:
        file_path = self.base_path / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)

    def _write_gcs(self, users: List[Dict], filename: str) -> None:
        blob_path = f"{self.gcs_prefix}/{filename}"
        blob = self.bucket.blob(blob_path)
        blob.upload_from_string(
            json.dumps(users, ensure_ascii=False),
            content_type="application/json"
        )
