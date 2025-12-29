import os
import yaml
from pathlib import Path
from src.utils.common import read_yaml
from src.constants import CONFIG_FILE_PATH
from src.config.config_entities import JobIngestionConfig


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH):
    
        self.config=read_yaml(config_filepath)

    def get_job_ingestion_config(self) -> JobIngestionConfig:
        config = self.config.job_ingestion_api

        return JobIngestionConfig(
            base_url=config.base_url,
            api_host=config.api_host,
            rate_limit_per_sec=config.rate_limit_per_sec,
            last_request_time=config.last_request_time,
            queries=config.queries,
            locations=config.locations,
            remote_options=config.remote_options,
            total_jobs=config.total_jobs,
            jobs_per_page=config.jobs_per_page,
            job_local_file_name=config.job_local_file_name,
            writer_mode=config.writer_mode,
            gcs_bucket_name=config.gcs_bucket_name,
            gcs_prefix=config.gcs_prefix,
            job_base_path=config.job_base_path
        )

