import torch
import pandas as pd
from typing import Iterator, Dict
import numpy as np

class FeatureReader:
    def __init__(self, source_path: str, batch_size: int = 1024):
        self.source_path = source_path
        self.batch_size = batch_size

    def stream_batches(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Reads data in chunks to avoid OOM (Out of Memory) issues.
        Returns a dictionary of tensors mapped to model input names.
        """
        # Example: Reading from a Parquet export (common for bulk embedding)
        chunks = pd.read_parquet(self.source_path, chunksize=self.batch_size)
        
        for chunk in chunks:
            yield self._transform_to_tensors(chunk)

    def _transform_to_tensors(self, df: pd.DataFrame) -> Dict:
        job_ids = df['job_id'].values.tolist()
        
        # 1. Convert columns to numpy first for easy horizontal stacking
        # Example: 512 (tokens) + 512 (mask) + 135 (metadata) = 1159
        tokens = np.array(df['token_ids'].tolist()) 
        masks = np.array(df['mask'].tolist())
        metadata = np.array(df['metadata_features'].tolist())

        # 2. Glue them together horizontally
        combined = np.hstack([tokens, masks, metadata]) # Shape: [Batch, 1159]
        
        # 3. Final Check: Force the dimension safety
        if combined.shape[1] != 1159:
            raise ValueError(f"Feature mismatch! Expected 1159, got {combined.shape[1]}")

        return {
            "ids": job_ids,
            "tensors": {
                "job_input": torch.tensor(combined, dtype=torch.float32)
            }
        }