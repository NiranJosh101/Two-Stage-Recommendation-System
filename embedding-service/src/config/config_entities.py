from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class MLflowConfig:
    model_name: str
    model_version: str
    tracking_uri: str

@dataclass(frozen=True)
class DataConfig:
    source_path: Path
    batch_size: int

@dataclass(frozen=True)
class PineconeConfig:
    index_name: str
    dimension: int
    metric: str
    api_key: str  # We will pull this from env vars


@dataclass(frozen=True)
class RedisConfig:
    host: str
    port: int
    db: int
    decode_responses: bool