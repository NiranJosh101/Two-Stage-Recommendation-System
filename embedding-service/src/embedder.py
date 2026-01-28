import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

class JobEmbedder:
    def __init__(self, model: torch.nn.Module, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def compute(self, batch: Dict) -> Tuple[List[str], np.ndarray]:
        job_ids = batch["ids"]
        
        job_tensors = batch["tensors"]["job_input"].to(self.device)

      
        job_embeddings = self.model.job_tower(job_tensors)

        # Normalize (Ensures compatibility with the Dot Product training)
        normalized_embeddings = F.normalize(job_embeddings, p=2, dim=1)

        return job_ids, normalized_embeddings.cpu().numpy()