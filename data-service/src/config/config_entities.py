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
