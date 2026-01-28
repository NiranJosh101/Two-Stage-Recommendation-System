import logging
from typing import List, Dict
import numpy as np
from pinecone import Pinecone

logger = logging.getLogger(__name__)

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
        # 1. Verification Guiderail
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch! DB expects {self.dimension}, got {vectors.shape[1]}")

        # 2. Prepare Payload
        records = []
        for i in range(len(ids)):
            record = {
                "id": str(ids[i]),
                "values": vectors[i].tolist(), # DBs expect lists, not numpy
            }
            if metadata:
                record["metadata"] = metadata[i]
            records.append(record)

        # 3. Execution with Error Handling
        try:
            # Pinecone handles small batches (like your 1024) well. 
            # If batch > 1000, use a loop to chunk this even further.
            response = self.index.upsert(vectors=records)
            logger.info(f"Successfully upserted {response['upserted_count']} vectors.")
        except Exception as e:
            logger.error(f"Vector DB Upsert failed: {e}")
            raise