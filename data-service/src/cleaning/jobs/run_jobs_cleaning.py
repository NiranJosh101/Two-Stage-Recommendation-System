import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.cleaning.jobs.cleaner import JobCleaner
from src.config.config_manager import ConfigurationManager

from src.validation.loaders.raw_data_loader import load_jobs_raw
from src.utils.exception import RecommendationsystemDataServie
from src.utils.logging import logging




config_manager = ConfigurationManager()
job_ingestion_config = config_manager.get_job_ingestion_config()



RAW_JOBS_PATH = Path(job_ingestion_config.job_base_path)
CLEAN_JOBS_PATH = Path(job_ingestion_config.job_clean_path)



# def load_jobs_raw(path: Path) -> List[Dict[str, Any]]:
#     with path.open("r", encoding="utf-8") as f:
#         return json.load(f)


def write_jobs_clean(jobs: List[Dict[str, Any]], path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(jobs, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise RecommendationsystemDataServie(
            f"Failed to write cleaned jobs to {path}: {e}"
        ) from e



def deduplicate_jobs(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        seen = set()
        deduped = []

        for job in jobs:
            job_id = job["job_id"]
            if job_id not in seen:
                seen.add(job_id)
                deduped.append(job)

        return deduped
    except Exception as e:
        raise RecommendationsystemDataServie(
            f"Failed to deduplicate jobs: {e}"
        ) from e



def run_jobs_cleaning():
    try:
        logging.info("<=== Starting Jobs Cleaning ===>")
        print("Loading raw jobs...")
        jobs_raw = load_jobs_raw(RAW_JOBS_PATH)

        print(f"Loaded {len(jobs_raw)} jobs")

        cleaner = JobCleaner()
        jobs_cleaned = cleaner.clean_many(jobs_raw)

        logging.info("Deduplicating jobs...")
        print("Deduplicating jobs...")
        jobs_cleaned = deduplicate_jobs(jobs_cleaned)

        logging.info("Writing cleaned jobs to file...")
        print(f"Writing {len(jobs_cleaned)} cleaned jobs...")
        write_jobs_clean(jobs_cleaned, CLEAN_JOBS_PATH)

        logging.info("<=== Job cleaning complete.===>")
        print("Job cleaning complete.")
    except Exception as e:
        raise RecommendationsystemDataServie(f"Job cleaning failed: {e}") from e


if __name__ == "__main__":
    run_jobs_cleaning()
