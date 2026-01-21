from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass(frozen=True)
class ModelConfig:
    user_embedding_dim: int
    job_embedding_dim: int
    output_dim: int
    hidden_dims: List[int]

@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    temperature: float
    top_k: int
    min_recall_threshold: float
    checkpoint_path: Path
    val_size: float
    num_workers: int

@dataclass(frozen=True)
class MLFlowConfig:
    experiment_name: str
    model_name: str