import mlflow.xgboost
import xgboost as xgb
import pandas as pd
import logging
from typing import List

class RankerModel:
    def __init__(self, model_uri: str):
        """
        Loads the model from MLflow.
        model_uri can be: 
        - 'models:/model_name/version' (Model Registry)
        - 'runs:/run_id/model' (Specific Run)
        """
        try:
            # Load the model using the MLflow XGBoost flavor
            # This returns the underlying xgb.Booster object
            self.bst = mlflow.xgboost.load_model(model_uri)
            
            # Feature consistency
            self.feature_cols = ["skill_overlap_score", "experience_gap"]
            self.group_col = "user_id"
            logging.info(f"MLflow model loaded successfully from {model_uri}")
        except Exception as e:
            logging.error(f"Failed to load model from MLflow: {e}")
            raise

    def predict(self, inference_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identical logic to before, but now running on an MLflow-sourced booster.
        """
        if inference_df.empty:
            return inference_df

        # 1. Feature selection (Matches training)
        X = inference_df[self.feature_cols]

        # 2. Grouping info (Critical for rank:ndcg)
        # We assume the Processor already sorted the DF by user_id
        group_counts = inference_df.groupby(self.group_col).size().to_list()

        # 3. Create DMatrix
        dtest = xgb.DMatrix(X)
        dtest.set_group(group_counts)

        # 4. Predict
        # MLflow's loaded booster works exactly like a standard xgb.Booster
        scores = self.bst.predict(dtest)

        # 5. Attach results
        results = inference_df.copy()
        results['score'] = scores

        # 6. Sort and return
        return results.sort_values(by='score', ascending=False).reset_index(drop=True)