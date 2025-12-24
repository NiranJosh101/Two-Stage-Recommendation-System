from user_generator import UserGenerator
from writer import UserWriter

from datetime import datetime

DEFAULT_NUM_USERS = 500


def run_users_ingestion(num_users: int = DEFAULT_NUM_USERS):
    """
    Orchestrates user ingestion:
    generate → persist (local for now)
    """

    generator = UserGenerator(seed=42)
    users = generator.generate(num_users)

    
    writer = UserWriter(
        mode="local",
        base_path="data/bronze/users"
    )
    
    filename = f"users_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    writer.write(users, filename)

    print(f"[Users Ingestion] Generated {len(users)} users")

    return {
        "entity": "users",
        "num_records": len(users)
    }


if __name__ == "__main__":
    run_users_ingestion()
