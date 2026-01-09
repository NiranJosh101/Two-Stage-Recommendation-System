import pandas as pd
import os

from src.cleaning.feature_transform.jobs.transformer import JobFeatureTransformer
from src.cleaning.feature_transform.users.transformer import UserFeatureTransformer
from src.utils.common import load_clean_data, write_jsonl

from src.config.config_manager import ConfigurationManager
from src.utils.exception import RecommendationsystemDataServie
from src.utils.logging import logging





def run_transform():
    """
    Main entrypoint to transform jobs and users into Feature Store-ready features.
    """
    config = ConfigurationManager()

    jobs_config = config.get_job_ingestion_config()
    users_config = config.get_user_data_ingestion_config()
    model_config = config.get_model_training_config()

    jobs = load_clean_data(jobs_config.job_clean_path)
    users = load_clean_data(users_config.user_clean_path)

    print(f"Loaded {len(jobs)} jobs and {len(users)} users for transformation.")

    job_transformer = JobFeatureTransformer()
    user_transformer = UserFeatureTransformer()

    jobs_fs = job_transformer.transform_many(jobs)
    users_fs = user_transformer.transform_many(users)

    # 🔹 Write FS outputs as JSONL
    write_jsonl(jobs_fs, model_config.job_feature_path)
    write_jsonl(users_fs, model_config.user_feature_path)

    print("✅ Transformation complete!")
    print(f"Jobs FS written to: {model_config.job_feature_path}")
    print(f"Users FS written to: {model_config.user_feature_path}")
