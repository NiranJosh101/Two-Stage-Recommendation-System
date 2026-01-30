import sys
import json
import redis 
from src.ingestion.users.user_generator import UserGenerator
from src.ingestion.users.writer import UserWriter
from src.config.config_manager import ConfigurationManager  
from src.utils.exception import RecommendationsystemDataServie
from src.utils.logging import logging 
from datetime import datetime, timezone

def hydrate_redis(users, redis_config):
    """
    Pushes raw user JSONs to Redis for millisecond hydration.
    """
    try:
        logging.info("Starting Redis hydration...")
        r = redis.Redis(
            host=redis_config.host, 
            port=redis_config.port, 
            db=redis_config.db,
            decode_responses=True
        )
        
        
        pipe = r.pipeline()
        for user in users:
            user_id = user.get("user_id")
            if user_id:
                key = f"user:raw:{user_id}"
                pipe.set(key, json.dumps(user))
        
        pipe.execute()
        logging.info(f"Successfully hydrated {len(users)} users to Redis.")
    except Exception as e:
        logging.error(f"Redis hydration failed: {e}")
        raise e

def run_users_ingestion():
    try:
        logging.info("<----- Starting Users Ingestion ----->")
        config_manager = ConfigurationManager()
        user_config = config_manager.get_user_data_ingestion_config()
        
        redis_config = config_manager.get_redis_config()

      
        generator = UserGenerator(users_config=user_config, seed=user_config.random_seed)
        users = generator.generate()

        
        writer = UserWriter(
            mode=user_config.writer_mode,
            base_path=user_config.user_base_path,
            config=user_config
        )
        filename = f"users_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        writer.write(users, filename)

       
        hydrate_redis(users, redis_config)

        print(f"[Users Ingestion] Generated and Hydrated {len(users)} users")

        return {
            "entity": "users",
            "num_records": len(users),
            "storage": ["disk", "redis"]
        }
          
    except Exception as e:
      
        logging.error("Users Ingestion Failed.") 
        raise RecommendationsystemDataServie(e, sys)

if __name__ == "__main__":
    run_users_ingestion()