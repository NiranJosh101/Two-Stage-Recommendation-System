import yaml
import os
from pathlib import Path
from dotenv import load_dotenv
from configs.config_entities import EmbeddingConfig, PineconeConfig

load_dotenv()

class ConfigurationManager:
    def __init__(self, config_filepath: str = "config.yaml"):
        # Logic to find the config file
        if not os.path.exists(config_filepath):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            fallback_path = os.path.join(current_dir, "config.yaml")
            
            if os.path.exists(fallback_path):
                config_filepath = fallback_path
            else:
                raise FileNotFoundError(f"Config file not found at {config_filepath} or {fallback_path}")

        with open(config_filepath, "r") as f:
            self.config = yaml.safe_load(f)

    def get_embedding_config(self) -> EmbeddingConfig:
        """Extracts the embedding service URL from the root of the YAML"""
        return EmbeddingConfig(
            service_url=self.config['embedding_service_url']
        )

    def get_pinecone_config(self) -> PineconeConfig:
        """Extracts Pinecone details and injects API Key from environment"""
        config = self.config['pinecone']
        
        return PineconeConfig(
            index_name=config['index_name'],
            index_host=config['index_host'],
            top_k=config['top_k'],
            api_key=os.getenv("PINECONE_API_KEY")
        )