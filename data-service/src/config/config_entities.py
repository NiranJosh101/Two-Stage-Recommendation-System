from dataclasses import dataclass
from typing import Optional




@dataclass
class JobIngestionConfig:
    base_url: str
    api_host: str
    rate_limit_per_sec: float
    last_request_time: float
    queries: list[str]
    locations: list[Optional[str]]
    remote_options: list[Optional[bool]]
    total_jobs: int 
    jobs_per_page: int
    job_local_file_name: Optional[str] 
    job_base_path: Optional[str] 
    gcs_prefix: Optional[str] 
    gcs_bucket_name: Optional[str] 
    writer_mode: Optional[str] 



@dataclass
class UserDataIngestionConfig:
    user_gcs_prefix: Optional[str]
    user_gcs_bucket_name: Optional[str]
    user_local_file_name: Optional[str]
    user_base_path: Optional[str]
    experience_levels: list[str]
    education_levels: list[str]
    locations: list[Optional[str]]
    writer_mode: Optional[str]
    random_seed: Optional[int]
    num_users: Optional[int]


@dataclass
class InteractionIngestionConfig:
    interaction_gcs_prefix: Optional[str]
    interaction_gcs_bucket_name: Optional[str]
    interaction_local_file_name: Optional[str]
    interaction_base_path: Optional[str]
    writer_mode: Optional[str]
    interaction_per_user: Optional[int]
    interaction_events_type: Optional[list[str]]
    interaction_events_weights: Optional[list[float]]
    interaction_seed: Optional[int]