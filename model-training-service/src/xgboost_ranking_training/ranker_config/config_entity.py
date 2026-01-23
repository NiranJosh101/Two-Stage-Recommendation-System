from dataclasses import dataclass
from typing import List, Dict

@dataclass(frozen=True)
class ModelConfig:
    target: str
    group_id: str
    features: List[str]
    xgboost_params: Dict

@dataclass(frozen=True)
class TrainingConfig:
    num_rounds: int
    val_size: float
    early_stopping_rounds: int

@dataclass(frozen=True)
class MLflowConfig:
    experiment_name: str
    model_name: str