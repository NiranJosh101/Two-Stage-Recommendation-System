import xgboost as xgb
from src.inference.model_loader import ModelLoader

class ModelServer:
    def __init__(self, loader: ModelLoader):
        self.loader = loader

    def predict(self, df):
        model = self.loader.load()
        
        # XGBRanker needs the features and the group size
        features = df.drop(columns=["user_id", "job_id"])
        
        
        dmatrix = xgb.DMatrix(features)
        dmatrix.set_group([len(df)])
        
        scores = model.predict(dmatrix)
        df["score"] = scores
        
        
        return df.sort_values(by="score", ascending=False)