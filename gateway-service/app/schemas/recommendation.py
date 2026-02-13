# app/schemas/recommendation.py
from pydantic import BaseModel, Field
from typing import List, Optional

class UserFeatures(BaseModel):
    """
    The core feature set required by the Embedding Service (User Tower).
    This matches the specific structure of the incoming user data.
    """
    user_id: str
    primary_roles: List[str]
    skills: List[str]
    experience_level: str
    education_level: str
    location: str
    years_of_experience: float

class RecommendRequest(BaseModel):
    """
    The top-level request object received by the Gateway.
    'features' is optional to handle the Returning User vs. New User logic.
    """
    user_id: str
    features: Optional[UserFeatures] = None
    

class RecommendationResponse(BaseModel):
    """
    The final 'Fat' object returned to the client after Ranking and Hydration.
    """
    user_id: str
    recommendations: List[dict]  
    status: str = "success"