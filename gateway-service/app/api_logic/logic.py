from typing import List, Dict, Any
from app.schemas.recommendation import UserFeatures, RecommendRequest
from app.clients.retriveval_client import RetrievalClient
from app.clients.ranking_client import RankingClient
from app.service_db.user_service import RedisManager


retrieval_client = RetrievalClient()
ranking_client = RankingClient()
redis_manager = RedisManager()

async def recommend_jobs_logic(
    request: RecommendRequest, 
    redis_manager: RedisManager,
    retrieval_client: RetrievalClient,
    ranking_client: RankingClient
) -> List[Dict[str, Any]]:
    user_id = request.user_id
    user_features = request.features

   
    if user_features:
        import asyncio
        asyncio.create_task(redis_manager.save_user_features(user_id, user_features))
    else:
        user_features = await redis_manager.get_user_features(user_id)
        if not user_features:
            return []

    
    candidate_ids = await retrieval_client.get_candidate_ids(user_features)
    if not candidate_ids:
        return []

    
    return await ranking_client.get_ranked_results(
        user_id=user_id,
        job_ids=candidate_ids,
        features=user_features
    )