from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


@dataclass
class modelTrainingConfig:
    user_feature_path: Optional[str]
    job_feature_path: Optional[str]
    final_dataset_path: Optional[str]
    ranking_dataset_path: Optional[str]
    ranking_dataset_random_seed: Optional[int]
    ranking_ds_skill_overlap_range: Optional[list[int]]
    ranking_ds_experience_gap_range: Optional[list[int]]
    two_tower_dataset_path: Optional[str]
    embed_model_names: Optional[str]
    user_embedding_dim: Optional[int]
    job_embedding_dim: Optional[int]
    allowed_labels: Optional[list[int]]
    feast_repo_path: Optional[str]
    fs_writer_version: Optional[str]