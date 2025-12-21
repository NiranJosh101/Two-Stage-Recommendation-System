import requests
import random
from typing import List, Dict

class JSearchClient:
    BASE_URL = "https://jsearch.p.rapidapi.com/search"

    def __init__(self, api_key: str):
        self.headers = {
            'x-rapidapi-key': "3fd00bef0emshca77e9913642e16p182fefjsn8449049325e8",
            'x-rapidapi-host': "jsearch.p.rapidapi.com"
        }

    def fetch_jobs(
        self,
        query: str,
        page: int = 1,
        location: str | None = None,
        remote: bool | None = None,
        num_pages: int = 1
    ) -> List[Dict]:
        params = {
            "query": query,
            "page": page,
            "num_pages": num_pages
        }

        if location:
            params["location"] = location
        if remote is not None:
            params["remote_jobs"] = str(remote).lower()

        response = requests.get(self.BASE_URL, headers=self.headers, params=params)
        response.raise_for_status()

        return response.json().get("data", [])
