import json
from pathlib import Path
import os
import sys
from typing import List, Dict

from src.supervision.negative_sampling.sampler import sample_negatives
from src.supervision.negative_sampling.popularity import compute_job_popularity
from src.config.config_manager import ConfigurationManager
from src.utils.exception import RecommendationsystemDataServie
from src.utils.logging import logging


def run_negative_sampling(
    labeled_positives: List[Dict[str, str]],
    all_jobs: List[str],
    ratio: int,
    use_popularity: bool = False,
    seed: int = 42
) -> List[Dict[str, str]]:
    """
    Generate negative samples for the dataset.
    """
    try:
        if not isinstance(all_jobs, list):
            raise TypeError("all_jobs must be a list of job_id strings")

        if not all(isinstance(j, str) for j in all_jobs):
            raise TypeError("all_jobs must contain only job_id strings")

        if use_popularity:
            popularity_dict = compute_job_popularity(labeled_positives)
            print(" Popularity-weighted sampling enabled — sampler must support weights")

        negatives = sample_negatives(
            labeled_positives=labeled_positives,
            all_jobs=all_jobs,   
            ratio=ratio,
            seed=seed
        )

        return negatives
    except Exception as e:
        logging.error(f"Error during negative sampling: {e}")
        raise RecommendationsystemDataServie(e, sys)




if __name__ == "__main__":
    config = ConfigurationManager()
    interaction_config = config.get_interaction_ingestion_config()
    jobs_config = config.get_job_ingestion_config()
    model_config = config.get_model_training_config()

    LABELED_PATH = Path(interaction_config.interaction_positive_path)
    JOBS_PATH = Path(jobs_config.job_clean_path)
    OUTPUT_PATH = Path(model_config.two_tower_dataset_path)

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load labeled positives
    with open(LABELED_PATH, "r", encoding="utf-8") as f:
        labeled_positives = json.load(f)

    # Load jobs
    with open(JOBS_PATH, "r", encoding="utf-8") as f:
        jobs = json.load(f)
    all_jobs = [job["job_id"] for job in jobs]

    # Negative sampling
    negatives = run_negative_sampling(
        labeled_positives=labeled_positives,
        all_jobs=all_jobs,
        ratio=interaction_config.interaction_nagative_sampling_ratio,
        use_popularity=False,
        seed=interaction_config.interaction_negative_sampling_seed
    )

    # Build final dataset
    final_dataset = labeled_positives + negatives

    # Write final dataset as pretty JSON
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=2)

    logging.info(f"✔ Wrote {len(final_dataset)} rows → {OUTPUT_PATH}")
    print(f"✔ Wrote {len(final_dataset)} rows → {OUTPUT_PATH}")
