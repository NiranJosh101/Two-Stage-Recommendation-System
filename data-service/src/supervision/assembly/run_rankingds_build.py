from pathlib import Path
import json
import random


from src.utils.common import load_clean_data
from src.config.config_manager import ConfigurationManager
from src.utils.exception import RecommendationsystemDataServie
from src.utils.logging import logging


config=ConfigurationManager()
model_config=config.get_model_training_config()
interaction_config=config.get_interaction_ingestion_config()


RANDOM_SEED = model_config.ranking_dataset_random_seed

INPUT_PATH = interaction_config.interaction_labeled_path
OUTPUT_PATH = Path(model_config.ranking_dataset_path)

SKILL_OVERLAP_RANGE = tuple(model_config.ranking_ds_skill_overlap_range)
EXPERIENCE_GAP_RANGE = tuple(model_config.ranking_ds_experience_gap_range)
 




def build_ranking_records(records: list[dict]) -> list[dict]:
    """
    Given a list of records with user_id, job_id, and label,
    generate minimal cross features for ranking.
    """

    random.seed(RANDOM_SEED)

    output = []

    for r in records:
        if not {"user_id", "job_id", "label"}.issubset(r):
            raise ValueError(f"Invalid record schema: {r}")

        enriched = {
            "user_id": r["user_id"],
            "job_id": r["job_id"],
            "skill_overlap_score": random.uniform(*SKILL_OVERLAP_RANGE),
            "experience_gap": random.randint(
                EXPERIENCE_GAP_RANGE[0], EXPERIENCE_GAP_RANGE[1]
            ),
            "label": r["label"],
        }

        output.append(enriched)

    return output




def main() -> None:
    print(f"Loading labeled interactions from: {INPUT_PATH}")

    records = load_clean_data(INPUT_PATH)

    print(f"Building ranking dataset ({len(records)} records)...")
    ranking_records = build_ranking_records(records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for record in ranking_records:
            f.write(json.dumps(record) + "\n")

    print(f"Ranking dataset written to: {OUTPUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
