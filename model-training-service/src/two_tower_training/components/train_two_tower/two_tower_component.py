import argparse
from pathlib import Path
import os
import yaml
import torch
from torch.utils.data import DataLoader

from src.two_tower_training.src_retriever.two_tower_config.config_manager import two_tower_ConfigurationManager
from src.two_tower_training.src_retriever.two_tower.retriver_model_archi import TwoTowerModel
from src.two_tower_training.src_retriever.two_tower.retriver_training import TwoTowerTrainer
from src.two_tower_training.src_retriever.two_tower.schema_validation import TowerSchema 
from src.two_tower_training.src_retriever.two_tower.dataset import JobDataset
from src.two_tower_training.src_retriever.two_tower.data_load import prepare_data


def run_training(args):
    # 1. Initialize Configuration Manager
    config_manager = two_tower_ConfigurationManager(args.config)
    
    # Pull separate config entities
    model_config = config_manager.get_model_config()
    train_config = config_manager.get_training_config(args.checkpoint_path)
    mlflow_config = config_manager.get_mlflow_config()

    # 2. Data Preparation
    print(f"Fetching data from: {args.feature_store_uri}")
    os.makedirs("./data_splits", exist_ok=True)
    train_file, val_file = prepare_data(
        feature_store_uri=args.feature_store_uri,
        temp_dir="./data_splits",
        val_size=0.2
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Ensure the output directory exists (Handle string vs Path object)
    output_file = Path(args.checkpoint_path)
    os.makedirs(output_file.parent, exist_ok=True)

    # 3. Initialize Schemas & Model
    user_schema = TowerSchema("user_embedding", model_config.user_embedding_dim)
    job_schema = TowerSchema("job_embedding", model_config.job_embedding_dim)

    model = TwoTowerModel(
        user_dim=model_config.user_embedding_dim,
        job_dim=model_config.job_embedding_dim,
        output_dim=model_config.output_dim,
        hidden_dims=model_config.hidden_dims
    ).to(device)

    # 4. Datasets & Loaders
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

    # 5. Initialize Trainer
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

    # 6. Start Training
    trainer.train(
        epochs=train_config.epochs,
        val_dataloader=val_loader,
        k=train_config.top_k,
        min_recall_threshold=train_config.min_recall_threshold
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="two_tower_config/config.yaml")
    parser.add_argument("--feature-store-uri", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True) # Required for KFP
    
    args = parser.parse_args()
    run_training(args)