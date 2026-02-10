from pydantic import BaseModel
from typing import Optional

class RedisConfig(BaseModel):
    url: str
    ttl_seconds: int

class ServiceClientConfig(BaseModel):
    url: str
    timeout: float
    connect_timeout: float

class AppConfig(BaseModel):
    title: str
    host: str
    port: int

class MasterConfig(BaseModel):
    app: AppConfig
    redis: RedisConfig
    retrieval: ServiceClientConfig
    ranking: ServiceClientConfig
    global_timeout: float