import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupShuffleSplit

def prepare_data(
    feature_store_uri: str, 
    temp_dir: str = "./data_splits", 
    val_size: float = 0.2,
    group_col: str = "user_id"
):
    try:
        df = pd.read_parquet(feature_store_uri)
    except Exception as e:
        raise e

    os.makedirs(temp_dir, exist_ok=True)

    # Use GroupShuffleSplit to keep users intact
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
    
    # .split returns indices. We use the group_col to define the groups.
    train_idx, val_idx = next(gss.split(df, groups=df[group_col]))

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    # Save to Parquet
    train_path = os.path.join(temp_dir, "train_split.parquet")
    val_path = os.path.join(temp_dir, "val_split.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    return train_path, val_path