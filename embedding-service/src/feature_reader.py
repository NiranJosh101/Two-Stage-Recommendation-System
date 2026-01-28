import pandas as pd
import numpy as np
import torch
from typing import Dict, Iterator

class FeatureReader:
    def __init__(self, source_path: str, batch_size: int = 1024):
        self.source_path = source_path
        self.batch_size = batch_size

    def stream_batches(self) -> Iterator[Dict]:
        df = pd.read_parquet(self.source_path)
        
        for i in range(0, len(df), self.batch_size):
            chunk = df.iloc[i : i + self.batch_size]
            yield self._transform_to_tensors(chunk)

    def _transform_to_tensors(self, df: pd.DataFrame) -> Dict:
        job_ids = df['job_id'].values.tolist()
        
        # 1. Grab the column that actually exists in your data
        # We convert the list of lists into a clean 2D numpy array
        raw_features = np.array(df['job_embedding'].tolist()) 
        
        # 2. Dimension Check
        # In your previous error, we expected 1159. 
        # Check if raw_features.shape[1] is 1159.
        if raw_features.shape[1] != 1159:
             # Just a warning for now so we can see what the real number is
             print(f"Note: Input features have dimension {raw_features.shape[1]}")

        return {
            "ids": job_ids,
            "tensors": {
                # This goes into the model's job_tower
                "job_input": torch.tensor(raw_features, dtype=torch.float32)
            }
        }