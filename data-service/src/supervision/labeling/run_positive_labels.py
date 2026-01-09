from pathlib import Path
import sys
import pandas as pd

from src.config.config_manager import ConfigurationManager
from src.utils.exception import RecommendationsystemDataServie
from src.utils.logging import logging


def run_positive_labels(labeled_interactions_path, output_path) -> pd.DataFrame:
    try:
        labeled_interactions_path = Path(labeled_interactions_path)
        output_path = Path(output_path)

        logging.info(f"Reading labeled data from {labeled_interactions_path}")

 
        df = pd.read_json(labeled_interactions_path)

        if df.empty:
            raise ValueError("Labeled interactions file is empty")

        if "label" not in df.columns:
            raise ValueError("Missing required column: label")

        positives = df[df["label"] == 1].copy()

        if positives.empty:
            raise ValueError("No positive interactions found (label == 1)")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        
        positives.to_json(
            output_path,
            orient="records",
            indent=2
        )

        logging.info(f"Saved {len(positives)} positive interactions")

        return positives

    except Exception as e:
        logging.error("Failed to extract positive labels", exc_info=True)
        raise RecommendationsystemDataServie(e, sys)


def main():
    config = ConfigurationManager()
    interaction_config = config.get_interaction_ingestion_config()

    positives = run_positive_labels(
        labeled_interactions_path=interaction_config.interaction_labeled_path,
        output_path=interaction_config.interaction_positive_path,
    )

    print(f"[OK] Extracted {len(positives)} positive interactions")


if __name__ == "__main__":
    main()
