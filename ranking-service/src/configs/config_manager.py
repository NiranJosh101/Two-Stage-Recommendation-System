import yaml
from pathlib import Path
from src.configs.config_entities import RedisConfig, MLFlowConfig, AppConfig, RankingConfig

class ConfigManager:
    def __init__(self, config_filepath: str = "config.yaml"):
        with open(config_filepath, "r") as f:
            self.config = yaml.safe_load(f)

    def get_ranking_config(self) -> RankingConfig:
        return RankingConfig(
            redis=RedisConfig(**self.config['redis']),
            mlflow=MLFlowConfig(**self.config['mlflow']),
            app=AppConfig(**self.config['app'])
        )