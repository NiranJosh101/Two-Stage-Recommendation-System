import sys
from typing import Dict
import pandas as pd


from src.validation.loaders.raw_data_loader import load_jobs_raw
from src.validation.loaders.raw_data_loader import load_users_raw
from src.validation.loaders.raw_data_loader import load_interactions_raw


from src.validation.contracts.jobs_contract import JOBS_RAW_CONTRACT
from src.validation.contracts.users_contracts import USERS_RAW_CONTRACT
from src.validation.contracts.interactions_contract import INTERACTIONS_RAW_CONTRACT


from src.validation.validators.schema_validator import validate_schema, SchemaValidationError
from src.validation.validators.primary_key_validator import validate_primary_key, PrimaryKeyValidationError
from src.validation.validators.empty_dataset_validator import validate_non_empty, EmptyDatasetError
from src.validation.validators.referential_validator import validate_referential_integrity, ReferentialIntegrityError

from src.config.config_manager import ConfigurationManager
from src.utils.exception import RecommendationsystemDataServie  
from src.utils.logging import logging 



def run_validation() -> Dict[str, pd.DataFrame]:
    """
    Run Phase 1 validation using system configuration.
    """
    validated_data = {}
    config_manager = ConfigurationManager()

    try:
        
        logging.info("Starting jobs validation")
        job_ingestion_config = config_manager.get_job_ingestion_config()

        jobs_df = load_jobs_raw(job_ingestion_config.job_base_path)
        validate_non_empty(jobs_df, "jobs_raw")
        validate_schema(jobs_df, JOBS_RAW_CONTRACT, "jobs_raw")
        validate_primary_key(
            jobs_df,
            JOBS_RAW_CONTRACT["primary_key"],
            "jobs_raw"
        )

        validated_data["jobs_raw"] = jobs_df
        logging.info("Jobs validation passed")

     
        logging.info("Starting users validation")
        user_ingestion_config = config_manager.get_user_data_ingestion_config()

        users_df = load_users_raw(user_ingestion_config.user_base_path)
        validate_non_empty(users_df, "users_raw")
        validate_schema(users_df, USERS_RAW_CONTRACT, "users_raw")
        validate_primary_key(
            users_df,
            USERS_RAW_CONTRACT["primary_key"],
            "users_raw"
        )

        validated_data["users_raw"] = users_df
        logging.info("Users validation passed")

       
        logging.info("Starting interactions validation")
        interaction_ingestion_config = config_manager.get_interaction_ingestion_config()

        interactions_df = load_interactions_raw(
            interaction_ingestion_config.interaction_base_path
        )

        # Synthetic PK (expected for raw interactions)
        interactions_df["interaction_id"] = interactions_df.index.astype(str)

        validate_non_empty(interactions_df, "interactions_raw")
        validate_schema(
            interactions_df,
            INTERACTIONS_RAW_CONTRACT,
            "interactions_raw"
        )
        validate_primary_key(
            interactions_df,
            INTERACTIONS_RAW_CONTRACT["primary_key"],
            "interactions_raw"
        )
        validate_referential_integrity(
            interactions_df,
            users_df,
            jobs_df
        )

        validated_data["interactions_raw"] = interactions_df
        logging.info("Interactions validation passed")

        logging.info("<===== Validation passed =====>")
        return validated_data

    except (
        SchemaValidationError,
        PrimaryKeyValidationError,
        EmptyDatasetError,
        ReferentialIntegrityError,
        RecommendationsystemDataServie,
    ) as e:
        logging.error("Validation failed", exc_info=True)
        raise e
    

if __name__ == "__main__":
    run_validation()

