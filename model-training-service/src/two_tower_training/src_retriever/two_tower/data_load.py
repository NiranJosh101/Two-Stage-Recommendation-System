import pandas as pd
from sklearn.model_selection import train_test_split
import os


def prepare_data(
    feature_store_uri: str, 
    temp_dir: str = "./temp_data", 
    val_size: float = 0.2
):

    try:
        df = pd.read_parquet(feature_store_uri)
    except Exception as e:
       
        raise

   

   
    os.makedirs(temp_dir, exist_ok=True)

    
    train_df, val_df = train_test_split(
        df, 
        test_size=val_size, 
        random_state=42
    )

    train_path = os.path.join(temp_dir, "train_split.parquet")
    val_path = os.path.join(temp_dir, "val_split.parquet")

   
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

   
    
    return train_path, val_path