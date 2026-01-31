import yaml
import os
from pathlib import Path
from dotenv import load_dotenv
from src.config.config_entities import MLflowConfig, DataConfig, PineconeConfig, RedisConfig, modelTrainingConfig

load_dotenv()

class ConfigurationManager:
    def __init__(self, config_filepath: str = "config.yaml"):
        # 1. Check if the file exists at the path provided (relative to terminal)
        if not os.path.exists(config_filepath):
            # 2. Fallback: Look in the same folder as this script (src/config/)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            fallback_path = os.path.join(current_dir, "config.yaml")
            
            if os.path.exists(fallback_path):
                config_filepath = fallback_path
            else:
                # If both fail, the error message will now show the full attempted path
                raise FileNotFoundError(f"Config file not found at {config_filepath} or {fallback_path}")

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

    def get_redis_config(self) -> RedisConfig:
        """
        Extracts Redis connection details using standard dictionary access.
        Assumes the key in your yaml is 'redis_config'.
        """
        # Fixed: Changed from self.config.redis to dictionary access
        config = self.config['redis_config']

        return RedisConfig(
            host=config['host'],
            port=config['port'],
            db=config['db'],
            decode_responses=config.get('decode_responses', True)
        )
    

    def get_model_training_config(self) -> modelTrainingConfig:
        config = self.config['model_training'] # Or 'model_training_config' depending on your YAML key

        return modelTrainingConfig(
            user_feature_path=config['user_feature_path'],
            job_feature_path=config['job_feature_path'],
            final_dataset_path=config['final_dataset_path'],
            ranking_dataset_path=config['ranking_dataset_path'],
            ranking_dataset_random_seed=config['ranking_dataset_random_seed'],
            ranking_ds_skill_overlap_range=config['ranking_ds_skill_overlap_range'],
            ranking_ds_experience_gap_range=config['ranking_ds_experience_gap_range'],
            two_tower_dataset_path=config['two_tower_dataset_path'],
            embed_model_names=config['embed_model_names'],
            user_embedding_dim=config['user_embedding_dim'],
            job_embedding_dim=config['job_embedding_dim'],
            allowed_labels=config['allowed_labels'],
            feast_repo_path=config['feast_repo_path'],
            fs_writer_version=config['fs_writer_version']
        )