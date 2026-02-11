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
            
            feature_dict = json.loads(data)
            
            # Safeguard: Ensure user_id is in the dict even if it wasn't in the JSON
            if "user_id" not in feature_dict:
                feature_dict["user_id"] = user_id
                
            return UserFeatures(**feature_dict)
            
        except Exception as e:
            # Replace with proper logging in production
            print(f"Redis Lookup Error: {e}")
            return None

    async def save_user_features(self, user_id: str, features: UserFeatures):
        """
        Store features in Redis with a TTL.
        Using model_dump_json() for Pydantic v2 compatibility.
        """
        try:
            # Ensure the object being saved actually belongs to the user_id key
            features.user_id = user_id 
            
            await self.pool.setex(
                f"user_feat:{user_id}",
                self.config.ttl_seconds,
                features.model_dump_json() # Updated from .json()
            )
        except Exception as e:
            print(f"Redis Save Error: {e}")

    async def close(self):
        """Cleanly close the connection pool"""
        await self.pool.close()