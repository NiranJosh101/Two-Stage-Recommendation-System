import os
import yaml
from pathlib import Path
from src.utils.common import read_yaml
from src.constants import CONFIG_FILE_PATH
from src.config.config_entities import JobIngestionConfig, UserDataIngestionConfig, InteractionIngestionConfig


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
    
    def get_user_data_ingestion_config(self) -> UserDataIngestionConfig:
        config = self.config.user_data_ingestion

        return UserDataIngestionConfig(
            user_gcs_prefix=config.user_gcs_prefix,
            user_gcs_bucket_name=config.user_gcs_bucket_name,
            user_local_file_name=config.user_local_file_name,
            user_base_path=config.user_base_path,
            experience_levels=config.experience_levels,
            education_levels=config.education_levels,
            locations=config.locations,
            writer_mode=config.writer_mode,
            random_seed=config.random_seed,
            num_users=config.num_users
        )

    def get_interaction_ingestion_config(self) -> InteractionIngestionConfig:
        config = self.config.interaction_ingestion

        return InteractionIngestionConfig(
            interaction_gcs_prefix=config.interaction_gcs_prefix,
            interaction_gcs_bucket_name=config.interaction_gcs_bucket_name,
            interaction_local_file_name=config.interaction_local_file_name,
            interaction_base_path=config.interaction_base_path,
            writer_mode=config.writer_mode,
            interaction_per_user=config.interaction_per_user,
            interaction_events_type=config.interaction_events_type,
            interaction_events_weights=config.interaction_events_weights,
            interaction_seed=config.interaction_seed
        )