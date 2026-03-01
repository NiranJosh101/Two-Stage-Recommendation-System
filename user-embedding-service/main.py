import sys
import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, status
import mlflow
from pydantic import BaseModel

# --- ADDED OTEL IMPORTS ---
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
# --------------------------

from config.config_manager import ConfigurationManager
from utils.exception import RecommendationsystemDataServie
from src.processor import UserProcessor
from src.model_loader import UserModelLoader
from src.embedder import UserEmbedder

class UserEmbeddingRequest(BaseModel):
    user_id: str
    primary_roles: list[str]
    skills: list[str]
    experience_level: str
    education_level: str
    location: str
    years_of_experience: float

class EmbeddingResponse(BaseModel):
    user_id: str
    vector: list[float]

app = FastAPI(title="User Embedding Service - Pure Inference")

# --- AUTO-INSTRUMENTATION ---
# This automatically traces every incoming request to this service
FastAPIInstrumentor.instrument_app(app)

# This traces any outgoing HTTP calls your service might make (to MLflow, etc.)
RequestsInstrumentor().instrument()


services = {}

@app.on_event("startup")
async def startup_event():
    try:
        logging.info("Initializing Pure Inference Embedding Service...")
        config_manager = ConfigurationManager()
        ml_cfg = config_manager.get_mlflow_config()

        mlflow.set_tracking_uri(ml_cfg.tracking_uri)
        
        services["processor"] = UserProcessor()
        
        loader = UserModelLoader(
            model_name=ml_cfg.model_name, 
            stage=ml_cfg.model_version
        )
        user_model = loader.load_user_tower()
        services["embedder"] = UserEmbedder(model=user_model)
        
        logging.info(f"Model '{ml_cfg.model_name}' loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to start User Service: {e}")
        sys.exit(1)

@app.post("/embed/user", response_model=EmbeddingResponse)
async def get_user_embedding(request: UserEmbeddingRequest):
    try:
        # OTEL TRACE CONTEXT: 
        # The trace started at the Gateway will be picked up here automatically.
        user_data = request.model_dump()
        
        processed_features = services["processor"].preprocess(user_data)
        vector = services["embedder"].compute(processed_features)

        return EmbeddingResponse(
            user_id=request.user_id,
            vector=vector.tolist()
        )
        
    except Exception as e:
        logging.error(f"Embedding error: {e}")
        raise RecommendationsystemDataServie(e, sys)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "mode": "stateless"}