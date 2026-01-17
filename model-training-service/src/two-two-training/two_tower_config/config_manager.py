import yaml
from pathlib import Path
from two_tower_config.config_entity import ModelConfig, TrainingConfig, MLFlowConfig

class two_tower_ConfigurationManager:
    def __init__(self, config_filepath: str):
        with open(config_filepath, 'r') as f:
            self.config = yaml.safe_load(f)

    def get_model_config(self) -> ModelConfig:
        config = self.config.model
        return ModelConfig(
            user_embedding_dim=config.user_embedding_dim,
            job_embedding_dim=config.job_embedding_dim,
            output_dim=config.output_dim,
            hidden_dims=config.hidden_dims
        )

    def get_training_config(self, checkpoint_path: str) -> TrainingConfig:
        config = self.config.train
        return TrainingConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            temperature=config.temperature,
            top_k=config.top_k,
            min_recall_threshold=config.min_recall_threshold,
            checkpoint_path=Path(checkpoint_path),
            val_size=config.get('val_size', 0.2),
            num_workers=config.get('num_workers', 2)
        )

    def get_mlflow_config(self) -> MLFlowConfig:
        config = self.config.mlflow
        return MLFlowConfig(
            experiment_name=config.experiment_name,
            model_name=config.model_name
        )