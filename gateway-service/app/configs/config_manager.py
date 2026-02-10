import os
import yaml
from pathlib import Path
from app.configs.config_manager import MasterConfig, AppConfig, RedisConfig, ServiceClientConfig

class ConfigurationManager:
    def __init__(self, config_filepath: str = "config/config.yaml"):
        self.config_path = Path(config_filepath)
        self.config = self._read_yaml()

    def _read_yaml(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def get_master_config(self) -> MasterConfig:
        """
        Maps YAML values to Pydantic Entities. 
        You can use os.getenv() here to allow env var overrides.
        """
        return MasterConfig(
            app=AppConfig(**self.config['app']),
            redis=RedisConfig(
                url=os.getenv("REDIS_URL", self.config['redis']['url']),
                ttl_seconds=self.config['redis']['ttl_seconds']
            ),
            retrieval=ServiceClientConfig(
                url=os.getenv("RETRIEVAL_URL", self.config['services']['retrieval']['url']),
                timeout=self.config['services']['retrieval']['timeout'],
                connect_timeout=self.config['services']['retrieval']['connect_timeout']
            ),
            ranking=ServiceClientConfig(
                url=os.getenv("RANKING_URL", self.config['services']['ranking']['url']),
                timeout=self.config['services']['ranking']['timeout'],
                connect_timeout=self.config['services']['ranking']['connect_timeout']
            ),
            global_timeout=self.config['global']['http_client_timeout']
        )

# Global settings instance
settings = ConfigurationManager().get_master_config()