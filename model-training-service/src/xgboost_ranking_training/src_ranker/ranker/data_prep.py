import pandas as pd
import os
from sklearn.model_selection import GroupShadowTarget  # Or a manual group split
import numpy as np

def prepare_data(
    feature_store_uri: str, 
    temp_dir: str = "./temp_data", 
    val_size: float = 0.2,
    group_col: str = "user_id" # Critical for ranking
):
    try:
        df = pd.read_parquet(feature_store_uri)
    except Exception as e:
        raise e

    os.makedirs(temp_dir, exist_ok=True)

    # 1. Get unique users
    unique_users = df[group_col].unique()
    
    # 2. Shuffle and split the USERS, not the rows
    np.random.seed(42)
    np.random.shuffle(unique_users)
    
    split_idx = int(len(unique_users) * (1 - val_size))
    train_users = unique_users[:split_idx]
    val_users = unique_users[split_idx:]

    # 3. Create splits based on user membership
    train_df = df[df[group_col].isin(train_users)]
    val_df = df[df[group_col].isin(val_users)]

    # 4. Save to Parquet
    train_path = os.path.join(temp_dir, "train_split.parquet")
    val_path = os.path.join(temp_dir, "val_split.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    return train_path, val_path