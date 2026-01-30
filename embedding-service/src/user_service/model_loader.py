import torch
import mlflow.pytorch
import logging
import sys
from src.utils.exception import RecommendationsystemDataServie

class UserModelLoader:
    def __init__(self, model_name: str = "user_tower_two_stage", stage: str = "Production"):
        """
        Loads the User Tower model from MLflow Model Registry.
        """
        self.model_uri = f"models:/{model_name}/{stage}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_user_tower(self) -> torch.nn.Module:
        """
        Fetches, configures for inference, and returns the User Tower model.
        """
        logging.info(f"Attempting to fetch User Tower from MLflow: {self.model_uri}")
        
        try:
            # 1. Load from MLflow
            model = mlflow.pytorch.load_model(self.model_uri)
            
            # 2. Move to device (GPU/CPU)
            model.to(self.device)
            
            # 3. Set to Evaluation Mode (Disables Dropout/Batch Norm)
            model.eval()
            
            # 4. Freeze gradients (Optimization for Inference)
            for param in model.parameters():
                param.requires_grad = False
                
            logging.info(f"User Tower successfully loaded on {self.device}")
            return model
            
        except Exception as e:
            logging.error(f"MLflow model loading failed for {self.model_uri}")
            raise RecommendationsystemDataServie(e, sys)