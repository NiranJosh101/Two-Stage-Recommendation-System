import mlflow.xgboost
import os

class ModelLoader:
    def __init__(self, model_name: str, stage: str = "Production"):
        self.model_name = model_name
        self.stage = stage
        self.model_uri = f"models:/{self.model_name}/{self.stage}"
        self._model = None

    def load(self):
        """Lazy load the model to ensure it's only fetched when needed."""
        if self._model is None:
           
            print(f"Fetching model {self.model_name} from stage {self.stage}...")
            self._model = mlflow.xgboost.load_model(self.model_uri)
        return self._model