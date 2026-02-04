import httpx
from fastapi import HTTPException
from configs.config_manager import ConfigurationManager

async def get_user_vector(
    http_client: httpx.AsyncClient, 
    payload: dict
) -> list[float]:
    
    config = ConfigurationManager()
    settings = config.get_embedding_config()
    
    """
    Passes the incoming Gateway payload directly to the Embedding Service.
    The Retrieval Service doesn't care if it's a user_id or raw_features.
    """
    url = f"{settings.service_url}/embed/user"

    try:
      
        response = await http_client.post(url, json=payload, timeout=2.0)
        
        
        response.raise_for_status()
        
        data = response.json()
        
        return data["vector"]

    except httpx.HTTPStatusError as exc:
        
        raise HTTPException(
            status_code=exc.response.status_code, 
            detail=f"Embedding Error: {exc.response.text}"
        )
    except Exception:
       
        raise HTTPException(status_code=503, detail="Embedding Service Unavailable")