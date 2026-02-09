from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.feature_hydration.hydrator import FeatureHydrator
from src.cross_feature_engineering.processors import FeatureProcessor
from src.inference.model_server import ModelServer
from src.inference.model_loader import ModelLoader
from src.configs.config_manager import ConfigManager

from src.utils.logging import logging
from src.utils.exception import RecommendationsystemDataServie


config_manager = ConfigManager()
cfg = config_manager.get_ranking_config()

app = FastAPI(title="Job Ranking Service")


model_loader = ModelLoader(
    model_name=cfg.mlflow.model_name, 
    stage=cfg.mlflow.stage
)
model_server = ModelServer(loader=model_loader)
hydrator = FeatureHydrator(
    host=cfg.redis.host, 
    port=cfg.redis.port
)
processor = FeatureProcessor()

class RankRequest(BaseModel):
    user_id: str
    job_ids: List[str]
    user_features: Optional[Dict[str, Any]] = None

@app.post("/rank")
async def rank_jobs(request: RankRequest):
    try:
        # User Hydration
        if request.user_features:
            user_profile = request.user_features
            user_profile["user_id"] = request.user_id
        else:
            user_profile = hydrator.get_user_features(request.user_id)
        
        if not user_profile:
            return {"user_id": request.user_id, "results": "fallback", "job_ids": request.job_ids}

        # Item Hydration
        raw_jobs = hydrator.get_features_batch(request.job_ids)
        if not raw_jobs:
            raise HTTPException(status_code=404, detail="No job features in Redis")

    
        feature_df = processor.create_grouped_dataset(user_profile, raw_jobs)

       
        ranked_df = model_server.predict(feature_df)

      
        top_n_ids = ranked_df.head(cfg.app.top_n)["job_id"].tolist()
        final_results = [job for job in raw_jobs if job.get("job_id") in top_n_ids]

        return {"user_id": request.user_id, "results": final_results}

    except RecommendationsystemDataServie as e:
        logging.error(f"Data service error: {e}")
        raise HTTPException(status_code=503, detail="Data service unavailable")