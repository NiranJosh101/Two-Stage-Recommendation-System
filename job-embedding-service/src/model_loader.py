import torch
import mlflow.pytorch
import logging
from src.utils.exception import RecommendationsystemDataServie
import sys

class ModelLoader:
    def __init__(self, model_name: str, stage_or_version: str = "Production"):
        """
        :param model_name: The name registered in MLflow (e.g., 'job_recommender_v1')
        :param stage_or_version: Version number like '2' or stage like 'Production'
        """
        # Construction happens here, but MLflow URI must be set in the main script first
        self.model_uri = f"models:/{model_name}/{stage_or_version}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_model(self) -> torch.nn.Module:
        logging.info(f"Loading model from {self.model_uri}...")
        
        try:
            # Load the model using the dynamically built URI
            model = mlflow.pytorch.load_model(self.model_uri)
            
            model.to(self.device)
            model.eval()
            
            # Freeze parameters for inference
            for param in model.parameters():
                param.requires_grad = False
                
            logging.info(f"Model {self.model_uri} loaded successfully on {self.device}.")
            return model
            
        except Exception as e:
            logging.error(f"Failed to load model from MLflow: {e}")
            # Raise custom exception for consistent error handling
            raise RecommendationsystemDataServie(e, sys)