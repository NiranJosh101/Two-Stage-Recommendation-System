import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class JobDataset(Dataset):
    def __init__(self, data_path: str, is_eval: bool = False):
        """
        data_path: Path to parquet containing the embeddings and labels.
        is_eval: If True, ensures true_item_indices are processed for metrics.
        """
        self.df = pd.read_parquet(data_path)
        self.is_eval = is_eval

       
        if self.is_eval and "job_index" not in self.df.columns:
            print("[INFO] 'job_index' not found in data. Generating unique indices from 'job_id'...")
           
            unique_jobs = self.df["job_id"].unique()
            job_id_to_idx = {job_id: i for i, job_id in enumerate(unique_jobs)}
            self.df["job_index"] = self.df["job_id"].map(job_id_to_idx)
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

       
        if "user_features" in row and isinstance(row["user_features"], dict):
            user_emb_list = row["user_features"].get("user_embedding")
        else:
            
            user_emb_list = row.get("user_embedding")

        if user_emb_list is None:
            raise KeyError(f"Could not find 'user_embedding' or 'user_features' at index {idx}. Available columns: {list(self.df.columns)}")


        user_emb = torch.tensor(user_emb_list, dtype=torch.float32)
        job_emb = torch.tensor(row["job_embedding"], dtype=torch.float32)

        sample = {
            "user_embedding": user_emb,
            "job_embedding": job_emb,
            "label": torch.tensor(row.get("label", 1.0), dtype=torch.float32)
        }

       
        if self.is_eval:
            sample["true_item_indices"] = [int(row["job_index"])]

        return sample