from typing import List, Dict
import random


def compute_job_popularity(labeled_positives: List[Dict[str, str]]) -> Dict[str, int]:
    """
    Compute the number of positives per job.

    """
    popularity: Dict[str, int] = {}
    for row in labeled_positives:
        job_id = row['job_id']
        popularity[job_id] = popularity.get(job_id, 0) + 1
    return popularity


def sample_weighted_jobs(job_ids: List[str], weights: List[float], k: int, seed: int = 42) -> List[str]:
    """
    Sample k jobs from job_ids using the given weights.

    """
    random.seed(seed)
    if k >= len(job_ids):
        return job_ids.copy()

    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    sampled_jobs = random.choices(job_ids, weights=normalized_weights, k=k)
    return sampled_jobs
