import yaml
from pathlib import Path


class ConfigManager:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get(self, *keys, default=None):
        """
        Access nested config values safely.
        """
        value = self.config
        for key in keys:
            value = value.get(key, {})
        return value or default
