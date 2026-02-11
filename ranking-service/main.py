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
        
        if not request.user_features:
            logging.warning(f"No features provided for user {request.user_id}. Using fallback.")
            return {
                "user_id": request.user_id, 
                "results": "fallback", 
                "job_ids": request.job_ids
            }
        
       
        user_profile = request.user_features
        user_profile["user_id"] = request.user_id

       
        raw_jobs = await hydrator.get_job_features_batch(request.job_ids)
        
        if not raw_jobs:
           
            return {"user_id": request.user_id, "results": "fallback"}


        feature_df = processor.create_grouped_dataset(user_profile, raw_jobs)

        # Model Inference
        ranked_df = model_server.predict(feature_df)

        
        top_n_ids = ranked_df.head(cfg.app.top_n)["job_id"].tolist()
        
        
        job_lookup = {str(job.get("job_id")): job for job in raw_jobs}
        final_results = [job_lookup[str(jid)] for jid in top_n_ids if str(jid) in job_lookup]

        return {"user_id": request.user_id, "results": final_results}

    except Exception as e:
        logging.error(f"Ranking Error: {e}")
        raise HTTPException(status_code=500, detail="Ranking pipeline failed")