import json
import redis.asyncio as redis
from typing import Optional
from app.schemas.recommendation import UserFeatures
from app.configs.config_manager import ConfigurationManager

class RedisManager:
    def __init__(self):
        # Initialize config and grab the redis entity
        self.config = ConfigurationManager().get_master_config().redis
        
        # Create the connection pool using configured URL
        self.pool = redis.from_url(
            self.config.url, 
            encoding="utf-8", 
            decode_responses=True
        )

    async def get_user_features(self, user_id: str) -> Optional[UserFeatures]:
        """
        Fetch user features from Redis and convert back to our Pydantic model.
        """
        try:
            data = await self.pool.get(f"user_feat:{user_id}")
            if not data:
                return None
            
            # Convert JSON string back to dictionary then to Pydantic
            feature_dict = json.loads(data)
            return UserFeatures(**feature_dict)
            
        except Exception as e:
            # In production, replace print with a proper logger
            print(f"Redis Lookup Error: {e}")
            return None

    async def save_user_features(self, user_id: str, features: UserFeatures):
        """
        Store features in Redis with a TTL (Time-To-Live).
        TTL is now driven by config.yaml (redis.ttl_seconds).
        """
        try:
            await self.pool.setex(
                f"user_feat:{user_id}",
                self.config.ttl_seconds,
                features.json()
            )
        except Exception as e:
            print(f"Redis Save Error: {e}")

    async def close(self):
        """Cleanly close the connection pool"""
        await self.pool.close()