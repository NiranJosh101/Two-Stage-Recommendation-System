from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class RetrievalResponse(BaseModel):
    status: str
    jobs_ids: List[str]  
    detail: Optional[str] = None


class JobResult(BaseModel):
    """
    The structure of the job objects returned in your 'final_results' list.
    Add/Adjust fields based on what your FeatureHydrator pulls from Redis.
    """
    job_id: str
    title: Optional[str] = None
    company: Optional[str] = None
   
    
    class Config:
        extra = "allow" 

class RankingResponse(BaseModel):
    user_id: str
    results: List[Dict[str, Any]] 