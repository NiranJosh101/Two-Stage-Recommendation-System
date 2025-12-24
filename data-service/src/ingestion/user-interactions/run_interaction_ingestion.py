import json
from pathlib import Path
from datetime import datetime

from users_interaction_generator import InteractionGenerator
from writer import InteractionWriter



USERS_BRONZE_PATH = Path("data/bronze/users")
JOBS_BRONZE_PATH = Path("data/bronze/jobs")

DEFAULT_INTERACTIONS_PER_USER = 10


def _load_latest_json(path: Path) -> list:
    """
    Load the most recent JSON file from a directory.
    """
    files = sorted(path.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {path}")
    latest_file = files[-1]

    with open(latest_file, "r", encoding="utf-8") as f:
        return json.load(f)


def run_interactions_ingestion(
    interactions_per_user: int = DEFAULT_INTERACTIONS_PER_USER
):
    """
    Orchestrates interaction ingestion:
    load users + jobs → generate interactions → persist
    """

    users = _load_latest_json(USERS_BRONZE_PATH)
    jobs = _load_latest_json(JOBS_BRONZE_PATH)

    generator = InteractionGenerator(seed=42)

    interactions = generator.generate(
        users=users,
        jobs=jobs,
        interactions_per_user=interactions_per_user
    )

    
    filename = f"interactions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

    writer = InteractionWriter(
        mode="local",
        base_path="data/bronze/interactions"
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

    return {
        "entity": "interactions",
        "num_users": len(users),
        "num_jobs": len(jobs),
        "num_interactions": len(interactions)
    }


if __name__ == "__main__":
    run_interactions_ingestion()
