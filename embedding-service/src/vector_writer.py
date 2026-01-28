import logging
from typing import List, Dict
import numpy as np
from pinecone import Pinecone

from src.utils.logging import logging


class VectorWriter:
    def __init__(self, api_key: str, index_name: str, dimension: int = 256):
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        self.dimension = dimension

    def upsert_batch(self, ids: List[str], vectors: np.ndarray, metadata: List[Dict] = None):
        """
        Formats and sends vectors to the DB.
        :param ids: List of unique Job IDs.
        :param vectors: NumPy array of shape (Batch, 256).
        :param metadata: List of dicts (e.g., [{'title': 'Dev', 'loc': 'NY'}, ...]).
        """
     
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch! DB expects {self.dimension}, got {vectors.shape[1]}")

       
        records = []
        for i in range(len(ids)):
            record = {
                "id": str(ids[i]),
                "values": vectors[i].tolist(), 
            }
            if metadata:
                record["metadata"] = metadata[i]
            records.append(record)

      
        try:
            response = self.index.upsert(vectors=records)
            logging.info(f"Successfully upserted {response['upserted_count']} vectors.")
        except Exception as e:
            logging.error(f"Vector DB Upsert failed: {e}")
            raise