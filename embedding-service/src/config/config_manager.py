import yaml
import os
from pathlib import Path
from dotenv import load_dotenv
from src.config.config_entities import MLflowConfig, DataConfig, PineconeConfig

load_dotenv()

class ConfigurationManager:
    def __init__(self, config_filepath: str = "config.yaml"):
        with open(config_filepath, "r") as f:
            self.config = yaml.safe_load(f)

    def get_mlflow_config(self) -> MLflowConfig:
        config = self.config['mlflow_config']
        return MLflowConfig(
            model_name=config['model_name'],
            model_version=str(config['model_version']),
            tracking_uri=config['tracking_uri']
        )

    def get_data_config(self) -> DataConfig:
        config = self.config['data_config']
        return DataConfig(
            source_path=Path(config['source_path']),
            batch_size=config['batch_size']
        )

    def get_pinecone_config(self) -> PineconeConfig:
        config = self.config['pinecone_config']
        return PineconeConfig(
            index_name=config['index_name'],
            dimension=config['dimension'],
            metric=config['metric'],
            api_key=os.getenv("PINECONE_API_KEY")
        )