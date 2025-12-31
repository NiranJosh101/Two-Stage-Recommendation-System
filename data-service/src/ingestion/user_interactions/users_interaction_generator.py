import sys
import random
from datetime import datetime
from typing import List, Dict
from src.config.config_entities import InteractionIngestionConfig
from src.utils.exception import RecommendationsystemDataServie
from src.utils.logging import logging 

class InteractionGenerator:
    """
    Generates synthetic user-job interactions.

    """

    def __init__(self, interaction_config: InteractionIngestionConfig, seed: int ):

        self.interaction_config = interaction_config
        self.EVENT_TYPES = interaction_config.interaction_events_type
        self.EVENT_WEIGHTS = interaction_config.interaction_events_weights 

        if seed is not None:
            random.seed(seed)

    def generate(
        self,
        users: List[Dict],
        jobs: List[Dict],
        interactions_per_user: int
    ) -> List[Dict]:
        try:
    
            interactions: List[Dict] = []

            if not users or not jobs:
                return interactions

            for user in users:
                user_id = user["user_id"]

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
        except Exception as e:
            raise RecommendationsystemDataServie(e, sys)