from pinecone import Pinecone, ServerlessSpec
import logging

from src.utils.logging import logging

class IndexManager:
    def __init__(self, api_key: str, index_name: str):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name

    def ensure_index_exists(self, dimension: int = 256, metric: str = "dotproduct"):
        """
        Guaranties the index is ready for use.
        """
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            logging.info(f"Creating new Pinecone index: {self.index_name}...")
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud="aws", 
                    region="us-east-1" # Double check this matches your dashboard
                )
            )
            logging.info("Index created successfully.")
        else:
            logging.info(f"Index '{self.index_name}' already exists and is ready.")