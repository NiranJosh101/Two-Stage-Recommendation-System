import httpx
from typing import List, Optional
from app.schemas.recommendation import UserFeatures
from app.schemas.service_responese import RetrievalResponse
from app.configs.config_manager import ConfigurationManager

class RetrievalClient:
    def __init__(self):
        # Fetch the specific retrieval config entity
        self.cfg = ConfigurationManager().get_master_config().retrieval
        
        self.url = f"{self.cfg.url}/retrieve"
        self.timeout = httpx.Timeout(
            self.cfg.timeout, 
            connect=self.cfg.connect_timeout
        )

    async def get_candidate_ids(self, features: UserFeatures) -> List[str]:
        """
        Calls the Retrieval Service using configured timeouts and URL.
        """
        payload = features.dict()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(self.url, json=payload)
                response.raise_for_status()
                
                data = RetrievalResponse(**response.json())
                return data.item_ids if data.status == "success" else []

            except httpx.HTTPStatusError as e:
                print(f"Retrieval Service Error [{e.response.status_code}] at {self.url}")
                return []
            except Exception as e:
                print(f"Unexpected error in RetrievalClient: {e}")
                return []