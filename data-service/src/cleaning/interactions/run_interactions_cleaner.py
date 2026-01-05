import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.cleaning.interactions.cleaner import InteractionCleaner
from src.config.config_manager import ConfigurationManager

from src.validation.loaders.raw_data_loader import load_interactions_raw
from src.utils.exception import RecommendationsystemDataServie
from src.utils.logging import logging


config_manager = ConfigurationManager()
interaction_ingestion_config = config_manager.get_interaction_ingestion_config()


RAW_INTERACTIONS_PATH = Path(interaction_ingestion_config.interaction_base_path)
CLEAN_INTERACTIONS_PATH = Path(interaction_ingestion_config.interaction_clean_path)


def write_interactions_clean(
    interactions: List[Dict[str, Any]],
    path: Path
) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(interactions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise RecommendationsystemDataServie(
            f"Failed to write cleaned interactions to {path}: {e}"
        ) from e


def deduplicate_interactions(
    interactions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    try:
        seen = set()
        deduped = []

        for interaction in interactions:
            interaction_id = interaction["interaction_id"]
            if interaction_id not in seen:
                seen.add(interaction_id)
                deduped.append(interaction)

        return deduped
    except Exception as e:
        raise RecommendationsystemDataServie(
            f"Failed to deduplicate interactions: {e}"
        ) from e


def run_interactions_cleaning():
    try:
        logging.info("<=== Starting Interactions Cleaning ===>")
        print("Loading raw interactions...")

        interactions_raw = load_interactions_raw(RAW_INTERACTIONS_PATH)
        print(f"Loaded {len(interactions_raw)} interactions")

        cleaner = InteractionCleaner()
        interactions_cleaned = cleaner.clean_many(interactions_raw)

        logging.info("Deduplicating interactions...")
        print("Deduplicating interactions...")
        interactions_cleaned = deduplicate_interactions(interactions_cleaned)

        logging.info("Writing cleaned interactions...")
        print(f"Writing {len(interactions_cleaned)} cleaned interactions...")
        write_interactions_clean(interactions_cleaned, CLEAN_INTERACTIONS_PATH)

        logging.info("<=== Interaction cleaning complete. ===>")
        print("Interaction cleaning complete.")
    except Exception as e:
        raise RecommendationsystemDataServie(
            f"Interaction cleaning failed: {e}"
        ) from e


if __name__ == "__main__":
    run_interactions_cleaning()
