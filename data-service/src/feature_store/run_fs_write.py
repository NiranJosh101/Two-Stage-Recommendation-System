import sys
import pandas as pd
from pathlib import Path
from src.feature_store.fs_contract import (
    RANKING_FEATURE_PRIMARY_KEYS,
    RANKING_FEATURES_SCHEMA,
    USER_FEATURES_SCHEMA,
    JOB_FEATURES_SCHEMA,
    TRAINING_DATASET_SCHEMA,
    USER_FEATURE_PRIMARY_KEYS,
    JOB_FEATURE_PRIMARY_KEYS,
    TRAINING_DATASET_PRIMARY_KEYS,
)

from src.utils.common import load_json_to_df, flatten_embeddings 
from src.feature_store.fs_validate import validate_feature_group
from src.feature_store.fs_writer import write_feature_group_feast
from src.config.config_manager import ConfigurationManager
from src.utils.exception import RecommendationsystemDataServie
from src.utils.logging import logging


config = ConfigurationManager()
model_config = config.get_model_training_config()

FEAST_REPO_PATH = model_config.feast_repo_path
VERSION = model_config.fs_writer_version

USER_FEATURES_JSON = model_config.user_feature_path
JOB_FEATURES_JSON = model_config.job_feature_path
TRAINING_DATA_JSON = model_config.final_dataset_path
RANKING_DATA_JSON = model_config.ranking_dataset_path



def main():
    try:
        logging.info("Starting Feature Store Write Process")
        df_user = load_json_to_df(USER_FEATURES_JSON)
        df_job = load_json_to_df(JOB_FEATURES_JSON)
        df_training = load_json_to_df(TRAINING_DATA_JSON)
        df_ranking = load_json_to_df(RANKING_DATA_JSON)

    
        df_training = flatten_embeddings(df_training)
        

        
        validate_feature_group(df_user, USER_FEATURES_SCHEMA, USER_FEATURE_PRIMARY_KEYS)
        validate_feature_group(df_job, JOB_FEATURES_SCHEMA, JOB_FEATURE_PRIMARY_KEYS)
        validate_feature_group(df_training, TRAINING_DATASET_SCHEMA, TRAINING_DATASET_PRIMARY_KEYS)
        validate_feature_group(df_ranking, RANKING_FEATURES_SCHEMA, RANKING_FEATURE_PRIMARY_KEYS)

        
        write_feature_group_feast(df_user, "user_features", VERSION, FEAST_REPO_PATH)
        write_feature_group_feast(df_job, "job_features", VERSION, FEAST_REPO_PATH)
        write_feature_group_feast(df_training, "training_dataset", VERSION, FEAST_REPO_PATH)
        write_feature_group_feast(df_ranking, "ranking_features", VERSION, FEAST_REPO_PATH)



        logging.info("✔✔ Feature Store Write Process Completed Successfully")
        print("✔✔ All feature groups ingested successfully!")

    except Exception as e:
        raise RecommendationsystemDataServie(e, sys) from e

if __name__ == "__main__":
    main()
