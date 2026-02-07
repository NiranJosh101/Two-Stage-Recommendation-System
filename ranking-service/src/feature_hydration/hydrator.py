import redis
import json
import logging
from typing import List, Dict, Any, Optional

class FeatureHydrator:
    def __init__(self, host='localhost', port=6379):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        # Separate prefixes for clarity
        self.item_prefix = "item:features:"
        self.user_prefix = "user:features:"

    def get_user_features(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetches the user profile (skills, exp, etc.) from Redis.
        """
        key = f"{self.user_prefix}{user_id}"
        raw_data = self.client.get(key)
        
        if not raw_data:
            logging.error(f"User profile for {user_id} missing from Redis!")
            return None
            
        return json.loads(raw_data)

    def get_features_batch(self, item_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetches the list of job features for the candidates.
        """
        keys = [f"{self.item_prefix}{i}" for i in item_ids]
        raw_data = self.client.mget(keys)
        
        hydrated_items = []
        for item in raw_data:
            if item:
                hydrated_items.append(json.loads(item))
            else:
                logging.warning("Job Candidate ID missing in Redis feature store.")
                
        return hydrated_items