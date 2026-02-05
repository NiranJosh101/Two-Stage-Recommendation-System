import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import redis

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





def upload_to_redis(jobs: List[Dict[str, Any]]):
    try:
       
        r = redis.Redis(
            host='localhost', 
            port=6379, 
            password=None,  
            decode_responses=True
        )

       
        if not r.ping():
            raise ConnectionError("Could not connect to Redis server.")

        logging.info(f"Starting Redis upload for {len(jobs)} items...")
        print(f"Syncing {len(jobs)} items to Redis...")
        
        pipe = r.pipeline()
        for job in jobs:
            job_id = job["job_id"]
          
            pipe.set(f"item:features:{job_id}", json.dumps(job))
        
        pipe.execute()
        logging.info("Redis upload successful.")
        print("Successfully synced data to Redis.")

    except Exception as e:
        logging.error(f"Redis upload failed: {e}")
        raise RecommendationsystemDataServie(f"Redis sync failed: {e}")


def run_jobs_cleaning():
    try:
        logging.info("<=== Starting Jobs Cleaning ===>")
        jobs_raw = load_jobs_raw(RAW_JOBS_PATH)
        
        cleaner = JobCleaner()
        jobs_cleaned = cleaner.clean_many(jobs_raw)

        logging.info("Deduplicating jobs...")
        jobs_cleaned = deduplicate_jobs(jobs_cleaned)

       
        logging.info("Writing cleaned jobs to file...")
        write_jobs_clean(jobs_cleaned, CLEAN_JOBS_PATH)

       
        upload_to_redis(jobs_cleaned)

        logging.info("<=== Job cleaning and Redis sync complete.===>")
    except Exception as e:
        raise RecommendationsystemDataServie(f"Job cleaning failed: {e}") from e
    




if __name__ == "__main__":
    run_jobs_cleaning()



    