import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class JobDataset(Dataset):
    def __init__(self, data_path: str, is_eval: bool = False):
        """
        data_path: Path to parquet or csv containing the embeddings and labels.
        is_eval: If True, ensures true_item_indices are processed for metrics.
        """
        # Loading Parquet is usually fastest for large embeddings
        self.df = pd.read_parquet(data_path)
        self.is_eval = is_eval

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Extract Embeddings
        # We ensure they are cast to float32 for PyTorch compatibility
        user_emb = torch.tensor(row["user_embedding"], dtype=torch.float32)
        job_emb = torch.tensor(row["job_embedding"], dtype=torch.float32)

        sample = {
            "user_embedding": user_emb,
            "job_embedding": job_emb,
            "label": torch.tensor(row["label"], dtype=torch.float32)
        }

        # 2. Extract Evaluation Metadata
        if self.is_eval:
            # The metrics logic expects a list of relevant item indices for each user
            # We wrap the single positive index in a list
            sample["true_item_indices"] = [int(row["job_index"])]

        return sample