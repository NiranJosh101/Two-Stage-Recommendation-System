import httpx
from fastapi import FastAPI, HTTPException, Depends
from contextlib import asynccontextmanager

from app.schemas.recommendation import RecommendRequest, RecommendationResponse
from app.api_logic.logic import recommend_jobs_logic
from app.service_db.user_service import RedisManager
from app.configs.config_manager import ConfigurationManager


# Initialize Config Manager
config_manager = ConfigurationManager()
cfg = config_manager.get_master_config()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages shared resource pools using validated configuration.
    """
    # Initialize Redis Manager (now uses cfg.redis internal to its __init__)
    app.state.redis_manager = RedisManager()
    
    # Global HTTP client using the configured global timeout
    app.state.http_client = httpx.AsyncClient(
        timeout=cfg.global_timeout
    )
    
    print(f"{cfg.app.title} Started: Connections Pooled")
    yield
    
    # Clean Shutdown
    await app.state.redis_manager.close()
    await app.state.http_client.aclose()
    print(f"{cfg.app.title} Stopped: Connections Closed")

# App initialization using config entities
app = FastAPI(
    title=cfg.app.title,
    lifespan=lifespan
)

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendRequest):
    """
    The Primary Handoff. 
    Coordinates the Retrieval -> Ranking pipeline via logic.py.
    """
    try:
        results = await recommend_jobs_logic(request)
        
        # Consistent response format
        return {
            "user_id": request.user_id,
            "recommendations": results if results else [],
            "status": "success" if results else "no_results"
        }

    except Exception as e:
        # Log the specific error here in production
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