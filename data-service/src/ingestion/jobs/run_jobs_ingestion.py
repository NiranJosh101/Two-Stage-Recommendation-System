import random
from api_client import JSearchClient
from writer import JobWriter

RAPID_API_KEY = "3fd00bef0emshca77e9913642e16p182fejsn8449049325e8"

QUERIES = ["developer", "designer", "marketing", "sales", "data", "engineer", "product"]
LOCATIONS = [None, "remote", "USA", "UK", "Canada", "Germany"]
REMOTE_OPTIONS = [None, True, False]

TOTAL_JOBS = 200
JOBS_PER_PAGE = 25

def run():
    client = JSearchClient(RAPID_API_KEY)
    writer = JobWriter()

    all_jobs = []
    page = 1

    while len(all_jobs) < TOTAL_JOBS:
        query = random.choice(QUERIES)
        location = random.choice(LOCATIONS)
        remote = random.choice(REMOTE_OPTIONS)

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
            if len(all_jobs) >= TOTAL_JOBS:
                break
            all_jobs.append(job)

        print(f"Collected {len(all_jobs)} jobs")
        page += 1

    writer.write(all_jobs, "jsearch_jobs_raw.json")

if __name__ == "__main__":
    run()
