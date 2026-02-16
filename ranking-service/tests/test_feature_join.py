import pytest
from src.feature_hydration.hydrator import FeatureHydrator

@pytest.mark.asyncio
async def test_feature_join_logic(redis_client):
    """
    Integration: Verifies that User + Job features are fetched 
    from Redis and concatenated correctly.
    """
    # Seed Redis with dummy features
    await redis_client.set("user:u123:feats", "0.1,0.5,0.9")
    await redis_client.set("job:j456:feats", "0.2,0.4,0.6")
    
    hydrator = FeatureHydrator(redis_client)
    
    # Act: Join the features
    joined_row = await hydrator.get_enriched_features(user_id="u123", job_id="j456")
    
    # Assert: Length should be user_feats + job_feats
    assert len(joined_row) == 6
    assert joined_row[0] == 0.1
    assert joined_row[-1] == 0.6