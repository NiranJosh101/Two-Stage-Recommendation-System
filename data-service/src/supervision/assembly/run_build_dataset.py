import json
from typing import List, Dict
from pathlib import Path

from src.supervision.assembly.dataset_builder import build_contrastive_dataset
from src.config.config_manager import ConfigurationManager
from src.utils.exception import RecommendationsystemDataServie
from src.utils.logging import logging


def run_build_dataset(
    labeled_positives: List[Dict[str, str]],
    sampled_negatives: List[Dict[str, str]],
    seed: int = 42,
) -> List[Dict[str, str]]:
    """
    Assemble the final trainable dataset.

    """
    try:
        dataset = build_contrastive_dataset(
            labeled_positives=labeled_positives,
            sampled_negatives=sampled_negatives,
            seed=seed,
            shuffle=True,
        )

        return dataset
    except Exception as e:
        logging.error(f"Error in building dataset: {e}")
        raise RecommendationsystemDataServie from e


if __name__ == "__main__":
    config = ConfigurationManager()
    interaction_config = config.get_interaction_ingestion_config()
    model_config = config.get_model_training_config()

    LABELED_POS_PATH = interaction_config.interaction_positive_path
    NEGATIVES_PATH = interaction_config.negative_sample_path
    OUTPUT_PATH = Path(model_config.two_tower_dataset_path)

    
    with open(LABELED_POS_PATH, "r") as f:
        labeled_positives = json.load(f)

    
    with open(NEGATIVES_PATH, "r") as f:
        sampled_negatives = json.load(f)

    final_dataset = run_build_dataset(
        labeled_positives=labeled_positives,
        sampled_negatives=sampled_negatives,
        seed=42,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=2)

    logging.info(f"✔ Wrote {len(final_dataset)} training samples → {OUTPUT_PATH}")
    print(f"✔ Wrote {len(final_dataset)} training samples → {OUTPUT_PATH}")
