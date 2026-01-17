import pandas as pd
from sklearn.model_selection import train_test_split
import os


def prepare_data(
    feature_store_uri: str, 
    temp_dir: str = "./temp_data", 
    val_size: float = 0.2
):
    """
    1. Fetches data from the Feature Store (URI).
    2. Performs a stratified or random split.
    3. Saves splits to local parquet files.
    4. Returns the paths to be used by the Dataset class.
    """

    
    # In a real scenario, you'd use a Feature Store client here.
    # For now, we assume the URI points to a parquet/CSV dump.
    try:
        df = pd.read_parquet(feature_store_uri)
    except Exception as e:
       
        raise

   

    # Ensure output directory exists
    os.makedirs(temp_dir, exist_ok=True)

    # 2. Perform the split
    # Note: For recommendation, you might want to split by 'timestamp' 
    # or 'user_id' to avoid data leakage. Here we use a standard random split.
    train_df, val_df = train_test_split(
        df, 
        test_size=val_size, 
        random_state=42
    )

    train_path = os.path.join(temp_dir, "train_split.parquet")
    val_path = os.path.join(temp_dir, "val_split.parquet")

    # 3. Save to local disk (Fastest for the subsequent Dataset load)
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

   
    
    return train_path, val_path