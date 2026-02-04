from pinecone import Pinecone
from configs.config_manager import ConfigurationManager
from fastapi import HTTPException



config = ConfigurationManager()
settings = config.get_pinecone_config()
pc = Pinecone(api_key=settings.api_key)

async def search_similar_items(vector: list[float], top_k: int = 100) -> list[str]:
    """
    Queries Pinecone for the nearest neighbors of the user vector.
    """
    try:
        
        async with pc.IndexAsyncio(host=settings.index_host) as index:
            results = await index.query(
                vector=vector,
                top_k=top_k,
                include_values=False,
                include_metadata=False  
            )
            
           
            return [match["id"] for match in results["matches"]]

    except Exception as e:
        
        print(f"Pinecone Error: {e}")
        raise HTTPException(status_code=500, detail="Vector Database search failed")