import sys
from src.ingestion.users.user_generator import UserGenerator
from src.ingestion.users.writer import UserWriter
from src.config.config_manager import ConfigurationManager  

from src.utils.exception import RecommendationsystemDataServie
from src.utils.logging import logging 

from datetime import datetime


def run_users_ingestion():
    try:
        """
        Orchestrates user ingestion:
        generate → persist (config-driven)
        """
        logging.info("<----- Starting Users Ingestion ----->")
        config_manager = ConfigurationManager()
        user_config = config_manager.get_user_data_ingestion_config()

        generator = UserGenerator(users_config=user_config, seed=user_config.random_seed)
        users = generator.generate()

        writer = UserWriter(
            mode=user_config.writer_mode,
            base_path=user_config.user_base_path,
            config=user_config
        )

        filename = f"users_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        writer.write(users, filename)

        print(f"[Users Ingestion] Generated {len(users)} users")

        return {
            "entity": "users",
            "num_records": len(users)
        }
          
    except Exception as e:
        logging.info("<----- Users Ingestion Completed. Total users collected: %d ----->")
        raise RecommendationsystemDataServie(e, sys)

    

if __name__ == "__main__":
    run_users_ingestion()
