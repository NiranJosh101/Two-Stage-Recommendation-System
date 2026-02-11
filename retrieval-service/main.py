from pyexpat import features
import httpx
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager


from utils.exception import RecommendationsystemDataServie
from utils.logging import logging
from configs.config_manager import ConfigurationManager
from src.client_embedding import get_user_vector
from src.client_vector_db import search_similar_items
from src.post_processor import refine_results

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        """
        Handles startup and shutdown events. 
        Ideal for managing shared resource pools.
        """
        config = ConfigurationManager()
        settings = config.get_embedding_config()
        
        logging.info("Starting Retrieval Service...")
        http_client = httpx.AsyncClient(
            base_url=settings.service_url,
            timeout=2.0
        )
        app.state.http_client = http_client
        
        print("Retrieval Service Started: Connections Pooled")
        
        yield  
        
        
        await http_client.aclose()
        logging.info("HTTP Client closed")
        print("Retrieval Service Stopped: Connections Closed")



    except Exception as e:
        logging.error(f"Service failed to start: {e}")
        raise RecommendationsystemDataServie("Retrieval Service failed to start")



app = FastAPI(lifespan=lifespan)

@app.post("/retrieve")
async def retrieve(request: Request):
    try:
        """
        The main orchestration endpoint.
        Gateway -> Retrieval -> Embedding Service -> Vector DB -> Post-Processor -> Result
        """
    
        # payload = await request.json()
        payload = features.model_dump()
        
        vector = await get_user_vector(request.app.state.http_client, payload)

        candidate_ids = await search_similar_items(vector, top_k=100)
        
        final_items = refine_results(candidate_ids)
        
        return {
            "status": "success",
            "jobs_ids": final_items
        }
    
    except RecommendationsystemDataServie as rds_exc:
        logging.error(f"Retrieval Error: {rds_exc}")
        return {
            "status": "error",
            "detail": str(rds_exc)
        }

@app.get("/health")
async def health():
    return {"status": "alive"}