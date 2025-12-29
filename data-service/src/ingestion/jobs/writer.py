import json
import sys
from typing import List, Dict
from pathlib import Path
from google.cloud import storage
from src.config.config_entities import JobIngestionConfig

from src.utils.exception import RecommendationsystemDataServie
from src.utils.logging import logging 


from pathlib import Path
from typing import List, Dict
import json
from google.cloud import storage

class JobWriter:
    def __init__(self, config: JobIngestionConfig, mode: str | None = None):
        """
        config: JobIngestionConfig
        mode: Optional override for writer mode ("local" or "gcs").
        """
        try:
            self.mode = mode or config.writer_mode

            if self.mode == "local":
                self.base_path = Path(config.job_base_path)
                self.base_path.mkdir(parents=True, exist_ok=True)

            elif self.mode == "gcs":
                if not config.gcs_bucket_name:
                    raise ValueError("bucket_name required for GCS mode")

                self.client = storage.Client()
                self.bucket = self.client.bucket(config.gcs_bucket_name)
                self.gcs_prefix = config.gcs_prefix or ""

            else:
                raise ValueError(f"Unsupported writer mode: {self.mode}")
        except Exception as e:
            raise RecommendationsystemDataServie(e, sys)

    def write(self, jobs: List[Dict], filename: str) -> None:
        try:
            if self.mode == "local":
                self._write_local(jobs, filename)
            elif self.mode == "gcs":
                self._write_gcs(jobs, filename)

            logging.info("jobs written successfully using %s mode", self.mode)
        except Exception as e:
            raise RecommendationsystemDataServie(e, sys)

    def _write_local(self, jobs: List[Dict], filename: str) -> None:
        try:
            file_path = self.base_path / filename
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(jobs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise RecommendationsystemDataServie(e, sys)

    def _write_gcs(self, jobs: List[Dict], filename: str) -> None:
        try:
            blob_path = f"{self.gcs_prefix}/{filename}"
            blob = self.bucket.blob(blob_path)
            blob.upload_from_string(
                json.dumps(jobs, ensure_ascii=False),
                content_type="application/json"
            )
        except Exception as e:
            raise RecommendationsystemDataServie(e, sys)
