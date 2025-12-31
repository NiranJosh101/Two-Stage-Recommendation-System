import json
import sys
from pathlib import Path
from datetime import datetime
from src.config.config_manager import ConfigurationManager 
from src.ingestion.user_interactions.users_interaction_generator import InteractionGenerator
from src.ingestion.user_interactions.writer import InteractionWriter




from src.utils.exception import RecommendationsystemDataServie
from src.utils.logging import logging 





from pathlib import Path

def _load_latest_json(path: str | Path) -> list:
    try:
        """
        Load the most recent JSON file from a directory.
        """
        path = Path(path) 

        files = sorted(path.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No JSON files found in {path}")

        latest_file = files[-1]

        with open(latest_file, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as e:
        raise RecommendationsystemDataServie(e, sys)

def run_interactions_ingestion(
):
    try:
        """
        Orchestrates interaction ingestion:
        load users + jobs → generate interactions → persist
        """
        logging.info("<----- Starting Interactions Ingestion ----->")
        config_manager = ConfigurationManager()
        jobs_config = config_manager.get_job_ingestion_config()
        user_config = config_manager.get_user_data_ingestion_config()
        interaction_config = config_manager.get_interaction_ingestion_config()


        users = _load_latest_json(user_config.user_base_path)
        jobs = _load_latest_json(jobs_config.job_base_path)

        generator = InteractionGenerator(interaction_config, interaction_config.interaction_seed)

        interactions = generator.generate(
            users=users,
            jobs=jobs,
            interactions_per_user=interaction_config.interactions_per_user
        )

        
        filename = f"interactions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

        writer = InteractionWriter(
            mode=interaction_config.writer_mode,
            base_path=interaction_config.interaction_base_path,
            bucket_name=interaction_config.interaction_gcs_bucket_name,
            gcs_prefix=interaction_config.interaction_gcs_prefix
        )
        writer.write(interactions, filename)

        # --- GCS persistence (enable later) ---
        # writer = InteractionWriter(
        #     mode="gcs",
        #     bucket_name="your-temp-bucket",
        #     gcs_prefix="interactions/raw"
        # )
        # writer.write(interactions, filename)

        print(f"[Interactions Ingestion]")
        print(f"Users loaded: {len(users)}")
        print(f"Jobs loaded: {len(jobs)}")
        print(f"Interactions generated: {len(interactions)}")
        print(f"Saved to: data/bronze/interactions/{filename}")

        logging.info("<----- Interactions Ingestion Completed ----->")
        return {
            "entity": "interactions",
            "num_users": len(users),
            "num_jobs": len(jobs),
            "num_interactions": len(interactions)
        }
    except Exception as e:
        raise RecommendationsystemDataServie(e, sys)

if __name__ == "__main__":
    run_interactions_ingestion()
