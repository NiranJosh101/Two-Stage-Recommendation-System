import argparse
import os
import sys
import logging
import pandas as pd
import xgboost as xgb
from pathlib import Path

# Assuming your files 1-3 are imported as modules
from src.xgboost_ranking_training.src_ranker.ranker.data_prep import prepare_data
from src.xgboost_ranking_training.src_ranker.ranker.data_loader import prepare_ranking_data
from src.xgboost_ranking_training.src_ranker.ranker.metric import compute_metrics
from src.xgboost_ranking_training.ranker_config.config_manager import RankingConfigurationManager 
from src.utils.exception import RecommendationsystemDataServie
from src.utils.logging import logging
def run_ranking_training(args):
    try:
        logging.info("Starting XGBoost Ranking Model Training Component")
        
        # 1. Configuration Management
        config_manager = RankingConfigurationManager(args.config)
        model_config = config_manager.get_model_config()  # features, target, etc.
        train_config = config_manager.get_training_config()
        mlflow_config = config_manager.get_mlflow_config()

        # 2. Data Preparation (File 1 logic)
        logging.info(f"Fetching data from: {args.feature_store_uri}")
        os.makedirs("./data_splits", exist_ok=True)
        
        train_file, val_file = prepare_data(
            feature_store_uri=args.feature_store_uri,
            temp_dir="./data_splits",
            val_size=0.2,
            group_col=model_config.group_id
        )
        
        # 3. Load & Format for XGBoost (File 2 logic)
        logging.info("Formatting data for Listwise Ranking...")
        train_df = pd.read_parquet(train_file)
        val_df = pd.read_parquet(val_file)

        dtrain, _, _ = prepare_ranking_data(
            train_df, 
            model_config.target, 
            model_config.group_id, 
            model_config.features
        )
        dval, _, y_val = prepare_ranking_data(
            val_df, 
            model_config.target, 
            model_config.group_id, 
            model_config.features
        )

        # 4. MLflow Session & Training (File 4 logic)
        import mlflow
        mlflow.set_experiment(mlflow_config.experiment_name)
        
        with mlflow.start_run(run_name="xgboost_ranker_run"):
            logging.info("Starting XGBoost training...")
            
            # Log params
            mlflow.log_params(model_config.xgboost_params)
            
            bst = xgb.train(
                model_config.xgboost_params,
                dtrain,
                num_boost_round=train_config.num_rounds,
                evals=[(dtrain, 'train'), (dval, 'validation')],
                early_stopping_rounds=10,
                verbose_eval=True
            )

            # 5. Metrics Calculation (File 3 logic)
            logging.info("Calculating Ranking Metrics...")
            val_preds = bst.predict(dval)
            
            # Ensure we use sorted IDs for the metric calculation
            val_df_sorted = val_df.sort_values(by=model_config.group_id)
            metrics = compute_metrics(
                y_true=y_val, 
                y_pred=val_preds, 
                group_ids=val_df_sorted[model_config.group_id].values
            )
            
            mlflow.log_metrics(metrics)
            logging.info(f"Training Metrics: {metrics}")

            # 6. Save Artifacts
            output_file = Path(args.model_output_path)
            os.makedirs(output_file.parent, exist_ok=True)
            bst.save_model(str(output_file))
            
            mlflow.xgboost.log_model(bst, "model")
            logging.info(f"✅ Model saved at: {args.model_output_path}")

    except Exception as e:
        # Custom Exception handling as per your style
        raise RecommendationsystemDataServie(e, sys)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ranking_config.yaml")
    parser.add_argument("--feature-store-uri", type=str, required=True)
    parser.add_argument("--model-output-path", type=str, required=True) # KFP Path
    
    args = parser.parse_args()
    run_ranking_training(args)