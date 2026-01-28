import mlflow.pytorch
import torch
import logging

from src.utils.logging import logging

class ModelLoader:
    def __init__(self, model_name: str, stage_or_version: str = "Production"):
        """
        :param model_name: The name registered in MLflow (e.g., 'job_tower_two_stage')
        :param stage_or_version: Can be 'Staging', 'Production', or a version number like '1'
        """
        # URI format: "models:/<model_name>/<stage_or_version>"
        self.model_uri = f"models:/{model_name}/{stage_or_version}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_model(self) -> torch.nn.Module:
        logging.info(f"Loading model from {self.model_uri}...")
        
        try:
            
            model = mlflow.pytorch.load_model(self.model_uri)
            
            model.to(self.device)
            model.eval()
            
            
            for param in model.parameters():
                param.requires_grad = False
                
            logging.info("Model loaded and moved to inference mode.")
            return model
            
        except Exception as e:
            logging.error(f"Failed to load model from MLflow: {e}")
            raise