import pandas as pd
from pathlib import Path

from src.feature_store.fs_contract import (
    USER_FEATURES_SCHEMA,
    JOB_FEATURES_SCHEMA,
    TRAINING_DATASET_SCHEMA,
    USER_FEATURE_PRIMARY_KEYS,
    JOB_FEATURE_PRIMARY_KEYS,
    TRAINING_DATASET_PRIMARY_KEYS,
)
from src.feature_store.fs_validate import validate_feature_group
from src.feature_store.fs_writer import write_feature_group_feast


FEAST_REPO_PATH = r"C:\Users\USER\feast_repo"  # change to your repo
VERSION = "v1"


USER_FEATURES_JSON = Path("data/user_features.json")
JOB_FEATURES_JSON = Path("data/job_features.json")
TRAINING_DATA_JSON = Path("data/training_dataset.json")


def load_json_to_df(json_path: Path) -> pd.DataFrame:
    if not json_path.exists():
        raise FileNotFoundError(f"{json_path} not found")
    return pd.read_json(json_path, lines=True)



def main():
    
    df_user = load_json_to_df(USER_FEATURES_JSON)
    df_job = load_json_to_df(JOB_FEATURES_JSON)
    df_training = load_json_to_df(TRAINING_DATA_JSON)

    # 2️⃣ Validate
    validate_feature_group(df_user, USER_FEATURES_SCHEMA, USER_FEATURE_PRIMARY_KEYS)
    validate_feature_group(df_job, JOB_FEATURES_SCHEMA, JOB_FEATURE_PRIMARY_KEYS)
    validate_feature_group(df_training, TRAINING_DATASET_SCHEMA, TRAINING_DATASET_PRIMARY_KEYS)

    # 3️⃣ Write to Feast
    write_feature_group_feast(df_user, "user_features", VERSION, FEAST_REPO_PATH)
    write_feature_group_feast(df_job, "job_features", VERSION, FEAST_REPO_PATH)
    write_feature_group_feast(df_training, "training_dataset", VERSION, FEAST_REPO_PATH)

    print("✅ All feature groups ingested successfully!")



if __name__ == "__main__":
    main()
