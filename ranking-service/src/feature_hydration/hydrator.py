import redis
import json
import logging
from typing import List, Dict, Any

class FeatureHydrator:
    def __init__(self, host='localhost', port=6379):
       
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.prefix = "item:features:"

    def get_features_batch(self, item_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Takes a list of IDs and returns the full objects from Redis.
        """
        keys = [f"{self.prefix}{i}" for i in item_ids]
        
        
        raw_data = self.client.mget(keys)
        
       
        hydrated_items = []
        for item in raw_data:
            if item:
                hydrated_items.append(json.loads(item))
            else:
                logging.warning("Candidate ID found in Retrieval but missing in Redis feature store.")
                
        return hydrated_items