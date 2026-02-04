import httpx
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager

from configs.config_manager import ConfigurationManager
from src.client_embedding import get_user_vector
from src.client_vector_db import search_similar_items
from src.post_processor import refine_results

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events. 
    Ideal for managing shared resource pools.
    """
    config = ConfigurationManager()
    settings = config.get_embedding_config()
    
   
    http_client = httpx.AsyncClient(
        base_url=settings.service_url,
        timeout=2.0
    )
    app.state.http_client = http_client
    
    print("Retrieval Service Started: Connections Pooled")
    
    yield  
    
    
    await http_client.aclose()
    print("Retrieval Service Stopped: Connections Closed")

app = FastAPI(lifespan=lifespan)

@app.post("/retrieve")
async def retrieve(request: Request):
    """
    The main orchestration endpoint.
    Gateway -> Retrieval -> Embedding Service -> Vector DB -> Post-Processor -> Result
    """
  
    payload = await request.json()
    
    
    vector = await get_user_vector(request.app.state.http_client, payload)
    

    candidate_ids = await search_similar_items(vector, top_k=100)
    
    final_items = refine_results(candidate_ids)
    
    return {
        "status": "success",
        "item_ids": final_items
    }

@app.get("/health")
async def health():
    return {"status": "alive"}