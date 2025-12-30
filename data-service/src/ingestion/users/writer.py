# src/ingestion/users/writer.py

import json
from pathlib import Path
import sys
from typing import List, Dict

from google.cloud import storage

from src.config.config_entities import UserDataIngestionConfig  
from src.utils.exception import RecommendationsystemDataServie
from src.utils.logging import logging




class UserWriter:
    def __init__(self, mode: str, base_path: str, config: UserDataIngestionConfig):
        try:
            self.mode = mode

            if self.mode == "local":
                if not config.user_base_path:
                    raise ValueError("user_base_path required for local mode")

                self.base_path = Path(base_path)
                self.base_path.mkdir(parents=True, exist_ok=True)

            elif self.mode == "gcs":
                if not config.user_gcs_bucket_name:
                    raise ValueError("gcs_bucket_name required for GCS mode")

                self.client = storage.Client()
                self.bucket = self.client.bucket(config.user_gcs_bucket_name)
                self.gcs_prefix = config.user_gcs_prefix or "users/synthetic"

            else:
                raise ValueError(f"Unsupported writer mode: {self.mode}")
            
        except Exception as e:
            logging.error("<----- User Writer Initialization Failed ----->")
            raise RecommendationsystemDataServie(e, sys)
        
        
    def write(self, users: List[Dict], filename: str) -> None:
        try:
            if self.mode == "local":
                self._write_local(users, filename)
            elif self.mode == "gcs":
                self._write_gcs(users, filename)
        except Exception as e:
            raise RecommendationsystemDataServie(e, sys)
        

    def _write_local(self, users: List[Dict], filename: str) -> None:
        try:
            file_path = self.base_path / filename
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(users, f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise RecommendationsystemDataServie(e, sys)
        

    def _write_gcs(self, users: List[Dict], filename: str) -> None:
        try:
            blob_path = f"{self.gcs_prefix}/{filename}"
            blob = self.bucket.blob(blob_path)
            blob.upload_from_string(
                json.dumps(users, ensure_ascii=False),
                content_type="application/json"
            )
        except Exception as e:
            raise RecommendationsystemDataServie(e, sys)
