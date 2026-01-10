import json
import sys
import uuid
from pathlib import Path
from typing import Dict, List

from src.utils.common import load_json, write_json, load_clean_data
from src.supervision.assembly.final_dataset_assembly.schemas import validate_training_row

from src.config.config_manager import ConfigurationManager
from src.utils.exception import RecommendationsystemDataServie
from src.utils.logging import logging


def build_training_dataset(
    interactions: List[dict],
    user_features: List[dict],
    job_features: List[dict],
) -> List[dict]:
    
    try:
        logging.info("Building training dataset from features...")
        # O(1) lookup tables
        # Take everything except the id as features
        user_map: Dict[str, dict] = {
            u["user_id"]: {k: v for k, v in u.items() if k != "user_id"}
            for u in user_features
        }
        job_map: Dict[str, dict] = {
            j["job_id"]: {k: v for k, v in j.items() if k != "job_id"}
            for j in job_features
        }

        training_rows: List[dict] = []

        for row in interactions:
            user_id = row["user_id"]
            job_id = row["job_id"]
            label = row["label"]

            if user_id not in user_map:
                continue  # or log missing user
            if job_id not in job_map:
                continue  # or log missing job

            training_row = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "job_id": job_id,
                "user_features": user_map[user_id],
                "job_features": job_map[job_id],
                "label": label,
            }

            # 🔒 Enforce contract
            validate_training_row(training_row)

            training_rows.append(training_row)

        return training_rows
    
    except Exception as e:
        raise RecommendationsystemDataServie(e, sys)


def run_feature_build() -> None:
    try:
     
        logging.info("Starting final dataset assembly process...")
        config = ConfigurationManager()
        model_config = config.get_model_training_config()
        jobs_config = config.get_job_ingestion_config()
        users_config = config.get_user_data_ingestion_config()

        interactions_path = model_config.two_tower_dataset_path
        users_fs_path = model_config.user_feature_path
        jobs_fs_path = model_config.job_feature_path
        output_path = model_config.final_dataset_path

        print("📥 Loading datasets...")
        interactions = load_clean_data(interactions_path)
        users_fs = load_clean_data(users_fs_path)
        jobs_fs = load_clean_data(jobs_fs_path)

        print("🔗 Assembling training dataset...")
        training_data = build_training_dataset(
            interactions=interactions,
            user_features=users_fs,
            job_features=jobs_fs,
        )

        print("💾 Writing output...")
        write_json(training_data, output_path)

        print("✅ Dataset assembly complete")
        print(f"Rows: {len(training_data)}")
        print(f"Saved to: {output_path}")
        logging.info("Final dataset assembly process completed successfully.")

    except Exception as e:
        raise RecommendationsystemDataServie(e, sys)

if __name__ == "__main__":
    run_feature_build()
