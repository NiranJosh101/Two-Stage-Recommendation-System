from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class RedisConfig:
    host: str
    port: int

@dataclass(frozen=True)
class MLFlowConfig:
    model_name: str
    stage: str

@dataclass(frozen=True)
class AppConfig:
    port: int
    top_n: int

@dataclass(frozen=True)
class RankingConfig:
    redis: RedisConfig
    mlflow: MLFlowConfig
    app: AppConfig