import json
import sys
from pathlib import Path
from typing import List, Dict

from google.cloud import storage
from src.utils.exception import RecommendationsystemDataServie
from src.utils.logging import logging 


class InteractionWriter:
    def __init__(
        self,
        mode: str,
        base_path: str,
        bucket_name: str,
        gcs_prefix: str
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

    def write(self, interactions: List[Dict], filename: str) -> None:
        try:
            if self.mode == "local":
                self._write_local(interactions, filename)
            elif self.mode == "gcs":
                self._write_gcs(interactions, filename)
        except Exception as e:
            raise RecommendationsystemDataServie(e, sys)

    def _write_local(self, interactions: List[Dict], filename: str) -> None:
        try:
            file_path = self.base_path / filename
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(interactions, f, ensure_ascii=False, indent=2)
                logging.info(f"Interactions written to {file_path}")
        except Exception as e:
            raise RecommendationsystemDataServie(e, sys)
        
    def _write_gcs(self, interactions: List[Dict], filename: str) -> None:
        try:
            blob_path = f"{self.gcs_prefix}/{filename}"
            blob = self.bucket.blob(blob_path)
            blob.upload_from_string(
                json.dumps(interactions, ensure_ascii=False),
                content_type="application/json"
            )
        except Exception as e:
            raise RecommendationsystemDataServie(e, sys)
