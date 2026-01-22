import argparse
from pathlib import Path
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader

from src.two_tower_training.src_retriever.two_tower_config.config_manager import two_tower_ConfigurationManager
from src.two_tower_training.src_retriever.two_tower.retriver_model_archi import TwoTowerModel
from src.two_tower_training.src_retriever.two_tower.retriver_training import TwoTowerTrainer
from src.two_tower_training.src_retriever.two_tower.schema_validation import TowerSchema 
from src.two_tower_training.src_retriever.two_tower.dataset import JobDataset
from src.two_tower_training.src_retriever.two_tower.data_load import prepare_data
from src.utils.exception import RecommendationsystemDataServie
from src.utils.logging import logging

def run_training(args):
    try:
        logging.info("Starting Two-Tower Model Training Component")
        config_manager = two_tower_ConfigurationManager(args.config)
        
    
        model_config = config_manager.get_model_config()
        train_config = config_manager.get_training_config(args.checkpoint_path)
        mlflow_config = config_manager.get_mlflow_config()
        
    
        print(f"Fetching data from: {args.feature_store_uri}")
        os.makedirs("./data_splits", exist_ok=True)
        logging.info("Preparing data...")
        train_file, val_file = prepare_data(
            feature_store_uri=args.feature_store_uri,
            temp_dir="./data_splits",
            val_size=0.2
        )
        logging.info(f"Data prepared. Train file: {train_file}, Validation file: {val_file}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        
        output_file = Path(args.checkpoint_path)
        os.makedirs(output_file.parent, exist_ok=True)

    
        user_schema = TowerSchema("user_embedding", model_config.user_embedding_dim)
        job_schema = TowerSchema("job_embedding", model_config.job_embedding_dim)
        logging.info("Schemas defined.")

        model = TwoTowerModel(
            user_dim=model_config.user_embedding_dim,
            job_dim=model_config.job_embedding_dim,
            output_dim=model_config.output_dim,
            hidden_dims=model_config.hidden_dims
        ).to(device)
        logging.info("Model initialized.")

        
        train_loader = DataLoader(
            JobDataset(train_file, is_eval=False), 
            batch_size=train_config.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            JobDataset(val_file, is_eval=True), 
            batch_size=train_config.batch_size, 
            shuffle=False
        )
        logging.info("DataLoaders created.")

        
        trainer = TwoTowerTrainer(
            model=model,
            user_schema=user_schema,
            job_schema=job_schema,
            train_dataloader=train_loader,
            learning_rate=train_config.learning_rate,
            temperature=train_config.temperature,
            device=device,
            checkpoint_path=str(output_file),
            experiment_name=mlflow_config.experiment_name,
            model_name=mlflow_config.model_name
        )
        logging.info("Starting training...")
        
        trainer.train(
            epochs=train_config.epochs,
            val_dataloader=val_loader,
            k=train_config.top_k,
            min_recall_threshold=train_config.min_recall_threshold
        )
        logging.info("Training completed successfully. Model saved at: {output_file}")
    except Exception as e:
        raise RecommendationsystemDataServie(e, sys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="two_tower_config/config.yaml")
    parser.add_argument("--feature-store-uri", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True) # Required for KFP
    
    args = parser.parse_args()
    run_training(args)