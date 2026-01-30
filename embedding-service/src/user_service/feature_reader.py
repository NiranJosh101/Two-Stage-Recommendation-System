import redis
import json
import sys
from src.config.config_manager import ConfigurationManager
from src.utils.exception import RecommendationsystemDataServie
from src.utils.logging import logging

class UserFeatureReader:
    def __init__(self):
        """
        Initializes the Redis client using the central configuration.
        """
        try:
            self.config_manager = ConfigurationManager()
            self.redis_cfg = self.config_manager.get_redis_config()
            
            self.client = redis.Redis(
                host=self.redis_cfg.host,
                port=self.redis_cfg.port,
                db=self.redis_cfg.db,
                decode_responses=self.redis_cfg.decode_responses
            )
            logging.info(f"UserFeatureReader connected to Redis at {self.redis_cfg.host}:{self.redis_cfg.port}")
        except Exception as e:
            raise RecommendationsystemDataServie(e, sys)

    def fetch_user_features(self, user_id: str) -> dict:
        """
        Retrieves the raw JSON profile from Redis for a given user_id.
        Returns None if the user does not exist.
        """
        try:
            key = f"user:raw:{user_id}"
            raw_data = self.client.get(key)
            
            if not raw_data:
                logging.warning(f"User ID {user_id} not found in Redis.")
                return None
            
            # Convert string back to dictionary for the Processor
            user_features = json.loads(raw_data)
            return user_features
            
        except Exception as e:
            logging.error(f"Error fetching features for user {user_id}")
            raise RecommendationsystemDataServie(e, sys)