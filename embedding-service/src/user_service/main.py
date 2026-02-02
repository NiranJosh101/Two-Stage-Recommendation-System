import sys
import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
import mlflow
from pydantic import BaseModel, Field



from src.config.config_manager import ConfigurationManager
from src.utils.exception import RecommendationsystemDataServie
from src.user_service.feature_reader import UserFeatureReader
from src.user_service.processor import UserProcessor
from src.user_service.model_loader import UserModelLoader
from src.user_service.embedder import UserEmbedder

# --- Schema Definitions ---

class UserRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="ID of a returning user to fetch from Redis")
    raw_features: Optional[Dict[str, Any]] = Field(None, description="Features for a new/cold-start user")

class EmbeddingResponse(BaseModel):
    user_id: str
    vector: list[float]
    source: str  # 'redis' or 'payload'



app = FastAPI(title="User Embedding Service")


services = {}

@app.on_event("startup")
async def startup_event():
    """Load models and initialize clients once at startup."""
    try:
        logging.info("Initializing User Embedding Service components with dynamic config...")
        
       
        config_manager = ConfigurationManager()
        ml_cfg = config_manager.get_mlflow_config()

        
        mlflow.set_tracking_uri(ml_cfg.tracking_uri)
        logging.info(f"MLflow Tracking URI set to: {ml_cfg.tracking_uri}")
        
        
        services["reader"] = UserFeatureReader()
        
       
        services["processor"] = UserProcessor()
        
       
        loader = UserModelLoader(
            model_name=ml_cfg.model_name, 
            stage=ml_cfg.model_version
        )
        user_model = loader.load_user_tower()
        
    
        services["embedder"] = UserEmbedder(model=user_model)
        
        logging.info(f"Service fully initialized. Model '{ml_cfg.model_name}' (v:{ml_cfg.model_version}) loaded.")
        
    except Exception as e:
        logging.error(f"Failed to start User Service: {e}")
       
        sys.exit(1)



@app.post("/embed/user", response_model=EmbeddingResponse)
async def get_user_embedding(request: UserRequest):
    """
    Dual-Input Endpoint:
    - If user_id is provided: Look up features in Redis.
    - If user_id is missing or not found: Use raw_features from payload.
    """
    user_data = None
    data_source = "unknown"

   
    if request.user_id:
        user_data = services["reader"].fetch_user_features(request.user_id)
        if user_data:
            data_source = "redis"

    # New User / Cold Start
    if not user_data and request.raw_features:
        user_data = request.raw_features
        # Ensure user_data has a user_id key even if temporary
        if "user_id" not in user_data:
            user_data["user_id"] = request.user_id or "cold_start_user"
        data_source = "payload"

    # Validation: Did we get data from either path?
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User features could not be found. Provide a valid user_id or raw_features."
        )

    try:
       
        processed_features = services["processor"].preprocess(user_data)
        vector = services["embedder"].compute(processed_features)

        return EmbeddingResponse(
            user_id=user_data["user_id"],
            vector=vector.tolist(),
            source=data_source
        )
        
    except Exception as e:
        raise RecommendationsystemDataServie(e, sys)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": "embedder" in services}