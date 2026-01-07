from typing import List, Dict, Set
import random


def sample_negatives(
    labeled_positives: List[Dict[str, str]],
    all_jobs: List[str],   # job IDs only
    ratio: int,
    seed: int
) -> List[Dict[str, str]]:
    """
    Generate negative samples for each user.
    """

    random.seed(seed)

    # user -> positive job_ids
    user_pos_jobs: Dict[str, Set[str]] = {}
    for row in labeled_positives:
        user_pos_jobs.setdefault(row["user_id"], set()).add(row["job_id"])

    all_job_ids = set(all_jobs)
    print(type(all_jobs))
    print(all_jobs if isinstance(all_jobs, (str, dict)) else all_jobs[:5])


    negatives: List[Dict[str, str]] = []

    for user_id, pos_jobs in user_pos_jobs.items():

        candidate_jobs = list(all_job_ids - pos_jobs)

        num_negatives = min(len(candidate_jobs), len(pos_jobs) * ratio)

        sampled_jobs = random.sample(candidate_jobs, num_negatives)

        for job_id in sampled_jobs:
            negatives.append(
                {
                    "user_id": user_id,
                    "job_id": job_id,
                    "label": 0,
                }
            )

    return negatives
