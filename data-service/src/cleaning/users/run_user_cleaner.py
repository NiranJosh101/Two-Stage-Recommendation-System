import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.cleaning.users.cleaner import UserCleaner
from src.config.config_manager import ConfigurationManager

from src.validation.loaders.raw_data_loader import load_users_raw
from src.utils.exception import RecommendationsystemDataServie
from src.utils.logging import logging


config_manager = ConfigurationManager()
user_ingestion_config = config_manager.get_user_data_ingestion_config()


RAW_USERS_PATH = Path(user_ingestion_config.user_base_path)
CLEAN_USERS_PATH = Path(user_ingestion_config.user_clean_path)


def write_users_clean(users: List[Dict[str, Any]], path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise RecommendationsystemDataServie(
            f"Failed to write cleaned users to {path}: {e}"
        ) from e


def deduplicate_users(users: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        seen = set()
        deduped = []

        for user in users:
            user_id = user["user_id"]
            if user_id not in seen:
                seen.add(user_id)
                deduped.append(user)

        return deduped
    except Exception as e:
        raise RecommendationsystemDataServie(
            f"Failed to deduplicate users: {e}"
        ) from e


def run_users_cleaning():
    try:
        logging.info("<=== Starting Users Cleaning ===>")
        print("Loading raw users...")

        users_raw = load_users_raw(RAW_USERS_PATH)
        print(f"Loaded {len(users_raw)} users")

        cleaner = UserCleaner()
        users_cleaned = cleaner.clean_many(users_raw)

        logging.info("Deduplicating users...")
        print("Deduplicating users...")
        users_cleaned = deduplicate_users(users_cleaned)

        logging.info("Writing cleaned users to file...")
        print(f"Writing {len(users_cleaned)} cleaned users...")
        write_users_clean(users_cleaned, CLEAN_USERS_PATH)

        logging.info("<=== User cleaning complete. ===>")
        print("User cleaning complete.")
    except Exception as e:
        raise RecommendationsystemDataServie(
            f"User cleaning failed: {e}"
        ) from e


if __name__ == "__main__":
    run_users_cleaning()
