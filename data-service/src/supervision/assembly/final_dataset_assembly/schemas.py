import sys
from typing import Dict, List, Any
from src.config.config_manager import ConfigurationManager
from src.utils.exception import RecommendationsystemDataServie
from src.utils.logging import logging


config = ConfigurationManager()
model_config = config.get_model_training_config()

USER_EMBEDDING_DIM = model_config.user_embedding_dim
JOB_EMBEDDING_DIM = model_config.job_embedding_dim
ALLOWED_LABELS = set(model_config.allowed_labels)



def _validate_embedding(
    embedding: List[float],
    expected_dim: int,
    field_name: str,
) -> None:
    try:
        if not isinstance(embedding, list):
            raise TypeError(f"{field_name} must be a list")

        if len(embedding) != expected_dim:
            raise ValueError(
                f"{field_name} has dimension {len(embedding)}, expected {expected_dim}"
            )
        
    except Exception as e:
        raise RecommendationsystemDataServie(e, sys)


def _validate_label(label: int) -> None:
    try:
        if label not in ALLOWED_LABELS:
            raise ValueError(f"Invalid label: {label}. Expected one of {ALLOWED_LABELS}")
    except Exception as e:
        raise RecommendationsystemDataServie(e, sys)



def validate_training_row(row: Dict[str, Any]) -> None:
    """
    Validate a single hydrated training row.

    """
    try:
        required_top_level_fields = {
            "user_id",
            "job_id",
            "user_features",
            "job_features",
            "label",
        }

        missing = required_top_level_fields - row.keys()
        if missing:
            raise KeyError(f"Missing required top-level fields: {missing}")

    
        if not isinstance(row["user_id"], str):
            raise TypeError("user_id must be a string")

        if not isinstance(row["job_id"], str):
            raise TypeError("job_id must be a string")

        
        user_features = row["user_features"]
        job_features = row["job_features"]

        if not isinstance(user_features, dict):
            raise TypeError("user_features must be a dict")

        if not isinstance(job_features, dict):
            raise TypeError("job_features must be a dict")

    
        if "user_embedding" not in user_features:
            raise KeyError("user_features missing 'user_embedding'")

        if "job_embedding" not in job_features:
            raise KeyError("job_features missing 'job_embedding'")

        _validate_embedding(
            user_features["user_embedding"],
            USER_EMBEDDING_DIM,
            field_name="user_features.user_embedding",
        )

        _validate_embedding(
            job_features["job_embedding"],
            JOB_EMBEDDING_DIM,
            field_name="job_features.job_embedding",
        )


        _validate_label(row["label"])
        
        logging.info(f"Validated training row for user_id: {row['user_id']}, job_id: {row['job_id']}")
    except Exception as e:
        raise RecommendationsystemDataServie(e, sys)
