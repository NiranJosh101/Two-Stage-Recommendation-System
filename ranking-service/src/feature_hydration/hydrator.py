import redis.asyncio as redis # Switching to async to match your other services
import json
import logging
from typing import List, Dict, Any, Optional

class FeatureHydrator:
    def __init__(self, host='localhost', port=6379):
        # Using async redis to prevent blocking the Ranking Service's event loop
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.item_prefix = "item:features:"

    async def get_job_features_batch(self, job_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetches the list of job features for the candidates.
        Only jobs are fetched here; User features come from the Gateway payload.
        """
        if not job_ids:
            return []

        keys = [f"{self.item_prefix}{i}" for i in job_ids]
        
        try:
            raw_data = await self.client.mget(keys)
            
            hydrated_jobs = []
            for i, item in enumerate(raw_data):
                if item:
                    hydrated_jobs.append(json.loads(item))
                else:
                    # Log which specific job is missing
                    logging.warning(f"Job ID {job_ids[i]} missing in Redis feature store.")
                    
            return hydrated_jobs
        except Exception as e:
            logging.error(f"Redis MGET failed in Ranking Service: {e}")
            return []

    async def close(self):
        await self.client.close()