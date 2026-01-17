import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader

# Importing from our local src/two_tower package

from src_retriever.two_tower.retriver_model_archi import TwoTowerModel
from src_retriever.two_tower.retriver_training import TwoTowerTrainer
from src_retriever.two_tower.schema_validation import TowerSchema
from src_retriever.two_tower.dataset import JobDataset

def run_training(args):
    # 1. Load Configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Setup Device and Directories
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    
    print(f"Starting training on device: {device}")

    # 3. Initialize Schemas
    # These validate that our pre-computed vectors are 783 and 1159 respectively
    user_schema = TowerSchema(
        feature_name="user_embedding", 
        expected_dim=config['model']['user_embedding_dim']
    )
    job_schema = TowerSchema(
        feature_name="job_embedding", 
        expected_dim=config['model']['job_embedding_dim']
    )

    # 4. Initialize Model
    model = TwoTowerModel(
        user_dim=config['model']['user_embedding_dim'],
        job_dim=config['model']['job_embedding_dim'],
        output_dim=config['model']['output_dim'],
        hidden_dims=config['model']['hidden_dims']
    )

    # 5. Prepare DataLoaders
    # We use is_eval=True for the validation set to ensure 'true_item_indices' is present
    train_dataset = JobDataset(args.train_path, is_eval=False)
    val_dataset = JobDataset(args.val_path, is_eval=True)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['train']['batch_size'], 
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['train']['batch_size'],
        shuffle=False
    )

    # 6. Initialize and Run Trainer
    # Note: loss_fn (BCE) and MLflow tracking are handled inside the trainer
    trainer = TwoTowerTrainer(
        model=model,
        user_schema=user_schema,
        job_schema=job_schema,
        train_dataloader=train_loader,
        learning_rate=float(config['train']['learning_rate']),
        temperature=float(config.get('train', {}).get('temperature', 1.0)),
        device=device,
        checkpoint_path=args.checkpoint_path,
        experiment_name=config['mlflow']['experiment_name'],
        model_name=config['mlflow']['model_name']
    )

    # 7. Start Training & Evaluation
    # min_recall_threshold handles the auto-promotion to 'Production' in MLflow
    trainer.train(
        epochs=config['train']['epochs'],
        val_dataloader=val_loader,
        k=config['train']['top_k'],
        min_recall_threshold=config['train'].get('min_recall_threshold', 0.7)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Paths can be local or mapped from a cloud bucket (S3/GCS) in a pipeline
    parser.add_argument("--config", type=str, required=True, help="Path to two_tower.yaml")
    parser.add_argument("--train-path", type=str, required=True, help="Path to training parquet")
    parser.add_argument("--val-path", type=str, required=True, help="Path to validation parquet")
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoints/two_tower.pt")
    
    args = parser.parse_args()
    run_training(args)