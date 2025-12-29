import os
import sys
from typing import List, Dict, Optional
import requests
import time
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.config.config_entities import JobIngestionConfig
from src.utils.exception import RecommendationsystemDataServie
from src.utils.logging import logging 


load_dotenv()


class JSearchClient:
    try:

        def __init__(self, config: JobIngestionConfig):
            
            """
            rate_limit_per_sec: max requests per second
            """
            self.headers = {
                'x-rapidapi-key': os.getenv("RAPID_API_KEY"),
                'x-rapidapi-host': config.api_host
            }
            self.rate_limit_per_sec = config.rate_limit_per_sec
            self._last_request_time = config.last_request_time
            self.BASE_URL = config.base_url

        def _wait_for_rate_limit(self):
            """Simple rate limiter based on time.sleep"""
            now = time.time()
            elapsed = now - self._last_request_time
            min_interval = 1 / self.rate_limit_per_sec
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self._last_request_time = time.time()

        @retry(
            retry=retry_if_exception_type(requests.exceptions.RequestException),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            stop=stop_after_attempt(5)
        )
        def fetch_jobs(
            self,
            query: str,
            page: int = 1,
            location: Optional[str] = None,
            remote: Optional[bool] = None,
            num_pages: int = 1
        ) -> List[Dict]:
            self._wait_for_rate_limit()  

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
        
        logging.info("<-----API call successful----->")  



    except Exception as e:
        raise RecommendationsystemDataServie(e, sys)
