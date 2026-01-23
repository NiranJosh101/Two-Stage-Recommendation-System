import yaml
from box import ConfigBox
from pathlib import Path
from src.xgboost_ranking_training.ranker_config.config_entity import ModelConfig, TrainingConfig, MLflowConfig

class RankingConfigurationManager:
    def __init__(self, config_filepath: str):
        with open(config_filepath, 'r') as f:
            self.config = ConfigBox(yaml.safe_load(f))

    def get_model_config(self) -> ModelConfig:
        config = self.config.model_config
        return ModelConfig(
            target=config.target,
            group_id=config.group_id,
            features=config.features,
            xgboost_params=config.xgboost_params
        )

    def get_training_config(self) -> TrainingConfig:
        config = self.config.training_config
        return TrainingConfig(
            num_rounds=config.num_rounds,
            val_size=config.get('val_size', 0.2),
            early_stopping_rounds=config.get('early_stopping_rounds', 10)
        )

    def get_mlflow_config(self) -> MLflowConfig:
        config = self.config.mlflow_config
        return MLflowConfig(
            experiment_name=config.experiment_name,
            model_name=config.model_name
        )