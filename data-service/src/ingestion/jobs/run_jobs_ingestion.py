import os
import sys
import random
from dotenv import load_dotenv

from src.ingestion.jobs.api_client import JSearchClient
from src.ingestion.jobs.writer import JobWriter
from src.config.config_manager import ConfigurationManager  

from src.utils.exception import RecommendationsystemDataServie
from src.utils.logging import logging 


load_dotenv()

RAPID_API_KEY = os.getenv("RAPID_API_KEY")


def run():
    try:

        config_manager = ConfigurationManager()
        job_ingestion_config = config_manager.get_job_ingestion_config()

        client = JSearchClient(job_ingestion_config)
        writer = JobWriter(job_ingestion_config, mode=job_ingestion_config.writer_mode)


        # writer = JobWriter(
        #     mode="gcs",
        #     bucket_name="your-bronze-bucket",
        #     gcs_prefix="jobs/jsearch/ingest_date=2025-12-22"
        # )

        all_jobs = []
        page = 1

        while len(all_jobs) < job_ingestion_config.total_jobs:
            query = random.choice(job_ingestion_config.queries)
            location = random.choice(job_ingestion_config.locations)
            remote = random.choice(job_ingestion_config.remote_options)

            jobs = client.fetch_jobs(
                query=query,
                page=page,
                location=location,
                remote=remote
            )

            if not jobs:
                page += 1
                continue

            for job in jobs:
                if len(all_jobs) >= job_ingestion_config.total_jobs:
                    break
                all_jobs.append(job)

            logging.info("<-----Fetched %d jobs for query='%s', location='%s', remote='%s'----->", len(jobs), query, location, remote)
            logging.info(f"Collected {len(all_jobs)} jobs")
            print(f"Collected {len(all_jobs)} jobs")
            page += 1

    
        writer.write(all_jobs, job_ingestion_config.job_local_file_name)
        logging.info("<-----Job Ingestion Completed. Total jobs collected: %d----->", len(all_jobs))
    except Exception as e:
         raise RecommendationsystemDataServie(e, sys)
    


if __name__ == "__main__":
    run()
