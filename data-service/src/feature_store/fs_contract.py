from typing import Dict, Any


USER_FEATURES_SCHEMA: Dict[str, Any] = {
    "user_id": "string",                
    "user_embedding": "array<float>",   
}

USER_FEATURE_PRIMARY_KEYS = ["user_id"]



JOB_FEATURES_SCHEMA: Dict[str, Any] = {
    "job_id": "string",                 
    "job_embedding": "array<float>",    
}

JOB_FEATURE_PRIMARY_KEYS = ["job_id"]




TRAINING_DATASET_SCHEMA: Dict[str, Any] = {
    "id": "string",                     
    "user_id": "string",              
    "job_id": "string",                
    "user_embedding": "array<float>",
    "job_embedding": "array<float>",
    "label": "int",                 
}

TRAINING_DATASET_PRIMARY_KEYS = ["user_id", "job_id"]


RANKING_FEATURES_SCHEMA = {
    "user_id": str,
    "job_id": str,
    "skill_overlap_score": float,
    "experience_gap": int,
    "label": int,
}

RANKING_FEATURE_PRIMARY_KEYS = ["user_id", "job_id"]

