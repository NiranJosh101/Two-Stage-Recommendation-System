import httpx
from typing import List, Dict, Any
from app.schemas.recommendation import UserFeatures
from app.configs.config_manager import ConfigurationManager

class RankingClient:
    def __init__(self):
        # Fetch the specific ranking config entity
        self.cfg = ConfigurationManager().get_master_config().ranking
        
        self.url = f"{self.cfg.url}/rank"
        self.timeout = httpx.Timeout(
            self.cfg.timeout, 
            connect=self.cfg.connect_timeout
        )

    async def get_ranked_results(
        self, 
        user_id: str, 
        job_ids: List[str], 
        features: UserFeatures
    ) -> List[Dict[str, Any]]:
        """
        Calls the Ranking Service to re-score candidates.
        Payload is structured to match the downstream RankRequest schema.
        """
        payload = {
            "user_id": user_id,
            "job_ids": job_ids,
            "user_features": features.model_dump(exclude={"user_id"})
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(self.url, json=payload)
                response.raise_for_status()
                
                data = response.json()
                
                # Handling the explicit 'fallback' string from your Ranking service
                if data.get("results") == "fallback":
                    return [] 
                
                return data.get("results", [])

            except httpx.HTTPStatusError as e:
                print(f"Ranking Service HTTP Error: {e.response.status_code} at {self.url}")
                return []
            except Exception as e:
                print(f"Unexpected error in RankingClient: {e}")
                return []