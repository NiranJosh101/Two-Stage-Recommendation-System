import pytest
from unittest.mock import MagicMock, patch
from src.client_embedding import get_user_vector

@patch("src.retrieval.service.Pinecone")
def test_query_parsing_logic(mock_pinecone_class):
    """
    Unit Test: Ensures we correctly parse Pinecone's ScoredVector objects.
    """
    # 1. Setup: Create a fake Pinecone response
    mock_index = MagicMock()
    mock_pinecone_class.return_value.Index.return_value = mock_index
    
    # Simulate Pinecone's return format
    mock_index.query.return_value = {
        "matches": [
            {"id": "job_123", "score": 0.98},
            {"id": "job_456", "score": 0.82}
        ]
    }

    service = get_user_vector(api_key="fake_key", index_name="test-index")
    
    # 2. Act
    results = service.get_user_vector(vector=[0.1]*128, top_k=2)

    # 3. Assert
    assert len(results) == 2
    assert results[0]["job_id"] == "job_123"
    assert results[0]["score"] == 0.98