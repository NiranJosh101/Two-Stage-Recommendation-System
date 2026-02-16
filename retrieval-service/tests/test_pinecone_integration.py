import os
import time
from pinecone import Pinecone, ServerlessSpec

def test_pinecone_upsert_and_query_integration():
    """
    Integration: Real round-trip to a dev/test index.
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "ci-test-index"

    # 1. Cleanup/Setup
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)
    
    pc.create_index(
        name=index_name, 
        dimension=128, 
        metric="cosine", 
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    index = pc.Index(index_name)

    # 2. Upsert dummy data
    index.upsert([("test_job_1", [0.1]*128)])
    
    # Pinecone is eventually consistent; wait a few seconds for indexing
    time.sleep(5) 

    # 3. Query and Verify
    res = index.query(vector=[0.1]*128, top_k=1)
    assert res["matches"][0]["id"] == "test_job_1"