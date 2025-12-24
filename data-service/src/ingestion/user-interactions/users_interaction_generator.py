import random
from datetime import datetime
from typing import List, Dict


class InteractionGenerator:
    """
    Generates synthetic user-job interactions.

    """

    EVENT_TYPES = ["view", "click", "apply"]
    EVENT_WEIGHTS = [0.7, 0.2, 0.1]  # realistic engagement distribution

    def __init__(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)

    def generate(
        self,
        users: List[Dict],
        jobs: List[Dict],
        interactions_per_user: int = 10
    ) -> List[Dict]:
        """
        Generate raw interaction events.

        Args:
            users: Clean or raw users (must include user_id)
            jobs: Clean or raw jobs (must include job_id)
            interactions_per_user: Avg number of jobs each user interacts with
        """
        interactions: List[Dict] = []

        if not users or not jobs:
            return interactions

        for user in users:
            user_id = user["user_id"]

            # Sample a subset of jobs per user (sparsity)
            sampled_jobs = random.sample(
                jobs,
                k=min(interactions_per_user, len(jobs))
            )

            for job in sampled_jobs:
                job_id = job["job_id"]

                event_type = random.choices(
                    self.EVENT_TYPES,
                    weights=self.EVENT_WEIGHTS,
                    k=1
                )[0]

                interactions.append({
                    "user_id": user_id,
                    "job_id": job_id,
                    "event_type": event_type,
                    "timestamp": datetime.utcnow().isoformat()
                })

        return interactions
