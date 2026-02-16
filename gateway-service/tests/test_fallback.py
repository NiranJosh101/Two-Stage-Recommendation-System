import pytest
from unittest.mock import AsyncMock, patch
from app.api_logic.logic import recommend_jobs_logic

@pytest.mark.asyncio
async def test_gateway_fallback_on_retrieval_failure():
    """
    Unit Test: If Retrieval fails, the Gateway should trigger the Fallback.
    This prevents the entire app from going down if Pinecone is offline.
    """
    # 1. Setup: Mock services where Retrieval raises an exception
    mock_embed = AsyncMock()
    mock_retrieval = AsyncMock(side_effect=Exception("Pinecone Timeout"))
    mock_fallback = AsyncMock(return_value=[{"job_id": "popular_1", "score": 1.0}])

    gateway = recommend_jobs_logic(
        embedding_svc=mock_embed,
        retrieval_svc=mock_retrieval,
        fallback_svc=mock_fallback
    )

    # 2. Act: Call the main recommend endpoint
    recommendations = await gateway.get_recommendations(user_id="u123")

    # 3. Assert: Verify we got the fallback result instead of a 500 error
    assert len(recommendations) == 1
    assert recommendations[0]["job_id"] == "popular_1"
    mock_fallback.assert_called_once()