import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.feature_hydration.hydrator import FeatureHydrator
from src.cross_feature_engineering.processors import FeatureProcessor
from src.inference.model_server import ModelServer
from src.inference.model_loader import ModelLoader

app = FastAPI(title="Job Ranking Service")

model_loader = ModelLoader(model_name="job-ranker-xgboost", stage="Production")
model_server = ModelServer(loader=model_loader)
hydrator = FeatureHydrator(host="localhost", port=6379)
processor = FeatureProcessor()

class RankRequest(BaseModel):
    user_id: str
    job_ids: List[str]
    # NEW: Optional field for new users (cold start bypass)
    user_features: Optional[Dict[str, Any]] = None

@app.post("/rank")
async def rank_jobs(request: RankRequest):
    try:
        # STEP A: User Hydration (with bypass for cold start)
        if request.user_features:
            # If the gateway sent features, use them directly!
            user_profile = request.user_features
            user_profile["user_id"] = request.user_id  # Ensure ID is present
            logging.info(f"Using provided features for new user {request.user_id}")
        else:
            # Otherwise, look up in Redis for existing users
            user_profile = hydrator.get_user_features(request.user_id)
        
        if not user_profile:
            logging.warning(f"No features provided or found for {request.user_id}")
            return {
                "user_id": request.user_id, 
                "results": "fallback", 
                "job_ids": request.job_ids
            }

        # STEP B: Hydration - Fetch job metadata
        raw_jobs = hydrator.get_features_batch(request.job_ids)
        
        if not raw_jobs:
            raise HTTPException(status_code=404, detail="No job features found in Redis")

        # STEP C: Feature Engineering
        # Create the DataFrame + Cross-features (Skill overlap, etc.)
        feature_df = processor.create_grouped_dataset(user_profile, raw_jobs)

        # STEP D: Inference
        # Get scores and sort the DataFrame by the XGBoost output
        ranked_df = model_server.predict(feature_df)

        # STEP E: Final "Fat" Selection
        # Take top 10 IDs from the ranked results
        top_10_ids = ranked_df.head(10)["job_id"].tolist()
        
        # Filter our raw_jobs list to only include these top 10
        # (This avoids a second Redis hit!)
        final_results = [job for job in raw_jobs if job.get("job_id") in top_10_ids]

        # Return to Gateway
        return {
            "user_id": request.user_id,
            "results": final_results 
        }

    except Exception as e:
        logging.error(f"Ranking failed: {str(e)}")
        # Fallback: return original order if ranking fails
        return {
            "user_id": request.user_id, 
            "results": "error", 
            "fallback": request.job_ids
        }