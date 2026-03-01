import httpx
from fastapi import FastAPI, HTTPException, Depends
from contextlib import asynccontextmanager

# --- ADDED OTEL IMPORTS ---
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
# --------------------------

from app.schemas.recommendation import RecommendRequest, RecommendationResponse
from app.api_logic.logic import recommend_jobs_logic
from app.service_db.user_service import RedisManager
from app.configs.config_manager import ConfigurationManager
from app.clients.retriveval_client import RetrievalClient
from app.clients.ranking_client import RankingClient
from app.utils.logging import logging
from app.utils.exception import RecommendationsystemDataServie

config_manager = ConfigurationManager()
cfg = config_manager.get_master_config()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        """
        Manages shared resource pools using validated configuration.
        """
        logging.info(f"Starting {cfg.app.title} with config: {cfg.dict()}")
        app.state.redis_manager = RedisManager()
        app.state.retrieval_client = RetrievalClient()
        app.state.ranking_client = RankingClient()
        
        app.state.http_client = httpx.AsyncClient(
            timeout=cfg.global_timeout
        )
        
        print(f"{cfg.app.title} Started: Connections Pooled")
        yield
        
        
        await app.state.redis_manager.close()
        await app.state.http_client.aclose()
        print(f"{cfg.app.title} Stopped: Connections Closed")
        logging
    except Exception as e:
        print(f"Error during lifespan management: {e}")
        raise RecommendationsystemDataServie("Service initialization failed") from e

app = FastAPI(
    title=cfg.app.title,
    lifespan=lifespan
)

FastAPIInstrumentor.instrument_app(app)


RequestsInstrumentor().instrument()

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendRequest):
    """
    The Primary Handoff. 
    Coordinates the Retrieval -> Ranking pipeline via logic.py.
    """
    try:
        results = await recommend_jobs_logic(request, app.state.redis_manager, app.state.retrieval_client, app.state.ranking_client)
        
       
        return {
            "user_id": request.user_id,
            "recommendations": results if results else [],
            "status": "success" if results else "no_results"
        }

    except Exception as e:
        
        raise HTTPException(
            status_code=500, 
            detail=f"Recommendation pipeline failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": cfg.app.title,
        "version": "1.0.0"
    }