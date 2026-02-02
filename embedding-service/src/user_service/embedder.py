import torch
import torch.nn.functional as F
import numpy as np
import sys
from src.utils.exception import RecommendationsystemDataServie

class UserEmbedder:
    def __init__(self, model: torch.nn.Module, device: str = None):
        """
        :param model: The User Tower model loaded from MLflow
        :param device: Execution device (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def compute(self, features: np.ndarray) -> np.ndarray:
        """
        Takes the processed feature array and generates a normalized user vector.
        """
        try:
           
            tensor_input = torch.from_numpy(features).float().to(self.device).unsqueeze(0)

           
            user_embeddings = self.model.user_tower(tensor_input)

           
            normalized_embeddings = F.normalize(user_embeddings, p=2, dim=1)

            
            return normalized_embeddings.cpu().numpy().flatten()
            
        except Exception as e:
            raise RecommendationsystemDataServie(e, sys)