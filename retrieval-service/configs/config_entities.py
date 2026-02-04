from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class EmbeddingConfig:
    service_url: str

@dataclass(frozen=True)
class PineconeConfig:
    index_name: str
    index_host: str
    top_k: int
    api_key: Optional[str] = None