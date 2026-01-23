import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import os
import json
from src.xgboost_ranking_training.src_ranker.ranker.data_loader import prepare_ranking_data
from src.xgboost_ranking_training.src_ranker.ranker.metric import compute_metrics


def train_ranker(
    train_data_path: str, 
    val_data_path: str, 
    model_output_path: str,
    config: dict
):
    # Start MLflow tracking
    # If running in Kubeflow, the tracking URI is usually set via env vars
    mlflow.set_experiment(config.get("experiment_name", "ranking_model_experiment"))

    with mlflow.start_run():
        # 1. Load Data
        train_df = pd.read_parquet(train_data_path)
        val_df = pd.read_parquet(val_data_path)

        # 2. Extract Config & Log Params
        feature_cols = config.get("features", ["skill_overlap_score", "experience_gap"])
        target_col = config.get("target", "label")
        group_col = config.get("group_id", "user_id")
        xgb_params = config.get("xgboost_params", {"objective": "rank:ndcg", "eval_metric": "ndcg@5"})
        
        mlflow.log_params(xgb_params)
        mlflow.log_param("num_rounds", config.get("num_rounds", 100))
        mlflow.log_param("features_used", feature_cols)

        # 3. Prepare DMatrices
        dtrain, X_train, y_train = prepare_ranking_data(train_df, target_col, group_col, feature_cols)
        dval, X_val, y_val = prepare_ranking_data(val_df, target_col, group_col, feature_cols)

        # 4. Train with XGBoost
        eval_list = [(dtrain, 'train'), (dval, 'validation')]
        
        bst = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=config.get("num_rounds", 100),
            evals=eval_list,
            early_stopping_rounds=10,
            verbose_eval=True
        )

        # 5. Validation & Custom Metrics
        val_preds = bst.predict(dval)
        # Ensure we use the sorted IDs to match the sorted DMatrix
        val_df_sorted = val_df.sort_values(by=group_col)
        val_group_ids = val_df_sorted[group_col].values
        
        metrics = compute_metrics(y_val, val_preds, val_group_ids)
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        print(f"✅ Metrics logged: {metrics}")

        # 6. Save & Log Model
        # Save locally for Kubeflow artifact tracking
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        bst.save_model(model_output_path)
        
        # Log the model to the MLflow registry
        mlflow.xgboost.log_model(bst, artifact_path="model")
        
        print(f"🚀 Run complete. Model saved to {model_output_path} and logged to MLflow.")
        
        return metrics