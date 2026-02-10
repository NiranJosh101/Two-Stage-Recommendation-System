# app/schemas/service_responses.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# --- Retrieval Service Response ---
class RetrievalResponse(BaseModel):
    status: str
    item_ids: List[str]  # Matching your return {"item_ids": final_items}
    detail: Optional[str] = None

# --- Ranking Service Response ---
class JobResult(BaseModel):
    """
    The structure of the job objects returned in your 'final_results' list.
    Add/Adjust fields based on what your FeatureHydrator pulls from Redis.
    """
    job_id: str
    title: Optional[str] = None
    company: Optional[str] = None
    # Add other fields that come back from your 'raw_jobs'
    
    class Config:
        extra = "allow" # Allows extra fields from hydration without crashing

class RankingResponse(BaseModel):
    user_id: str
    results: List[Dict[str, Any]] # Or List[JobResult] for stricter validation