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



def run_validation(paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Run validation on Jobs, Users, and Interactions datasets independently.
    Raises:
        Any validator exception if validation fails
    """
    validated_data = {}

    
    try:
        config_manager = ConfigurationManager()
        job_ingestion_config = config_manager.get_job_ingestion_config()
        job_path = job_ingestion_config.job_base_path
        jobs_df = load_jobs_raw(job_path)
        validate_non_empty(jobs_df, dataset_name="jobs_raw")
        validate_schema(jobs_df, JOBS_RAW_CONTRACT, dataset_name="jobs_raw")
        validate_primary_key(jobs_df, JOBS_RAW_CONTRACT["primary_key"], dataset_name="jobs_raw")
        validated_data["jobs_raw"] = jobs_df
        print("JobsRaw validation passed")
    except (SchemaValidationError, PrimaryKeyValidationError, EmptyDatasetError) as e:
        print(f"JobsRaw validation failed\n{e}")
        raise

    
    try:
        config_manager = ConfigurationManager()
        user_ingestion_config = config_manager.get_user_data_ingestion_config()
        users_df = load_users_raw(user_ingestion_config.user_base_path)
        validate_non_empty(users_df, dataset_name="users_raw")
        validate_schema(users_df, USERS_RAW_CONTRACT, dataset_name="users_raw")
        validate_primary_key(users_df, USERS_RAW_CONTRACT["primary_key"], dataset_name="users_raw")
        validated_data["users_raw"] = users_df
        print("UsersRaw validation passed")
    except (SchemaValidationError, PrimaryKeyValidationError, EmptyDatasetError) as e:
        print(f"UsersRaw validation failed \n{e}")
        raise

    
    try:
        config_manager = ConfigurationManager()
        interaction_ingestion_config = config_manager.get_interaction_ingestion_config()
        interactions_df = load_interactions_raw(interaction_ingestion_config.interaction_base_path)
        # Generate synthetic PK if not already present
        if "interaction_id" not in interactions_df.columns:
            interactions_df["interaction_id"] = interactions_df.index.astype(str)

        validate_non_empty(interactions_df, dataset_name="interactions_raw")
        validate_schema(interactions_df, INTERACTIONS_RAW_CONTRACT, dataset_name="interactions_raw")
        validate_primary_key(interactions_df, INTERACTIONS_RAW_CONTRACT["primary_key"], dataset_name="interactions_raw")
        validate_referential_integrity(
            interactions_df=interactions_df,
            users_df=users_df,
            jobs_df=jobs_df,
            dataset_name="interactions_raw"
        )
        validated_data["interactions_raw"] = interactions_df
        print("InteractionsRaw validation passed")
    except (SchemaValidationError, PrimaryKeyValidationError, EmptyDatasetError, ReferentialIntegrityError) as e:
        print(f"InteractionsRaw validation failed\n{e}")
        raise

    print("All Phase 1 validation passed")
    return validated_data
