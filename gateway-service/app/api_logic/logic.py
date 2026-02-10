from typing import List, Dict, Any
from app.schemas.recommendation import UserFeatures, RecommendRequest
from app.clients.retriveval_client import RetrievalClient
from app.clients.ranking_client import RankingClient
from app.service_db.user_service import RedisManager

# Initialize components
retrieval_client = RetrievalClient()
ranking_client = RankingClient()
redis_manager = RedisManager()

async def recommend_jobs_logic(request: RecommendRequest) -> List[Dict[str, Any]]:
    """
    Orchestrates the flow: 
    1. Resolve Features -> 2. Retrieval -> 3. Ranking
    """
    user_id = request.user_id
    user_features = request.features

    # --- STEP 1: Feature Resolution (Freshness Logic) ---
    if user_features:
        # Case: New User or Client-provided fresh data
        # We save to Redis in the background to keep it fast
        await redis_manager.save_user_features(user_id, user_features)
    else:
        # Case: Returning User - fetch from our fast cache
        user_features = await redis_manager.get_user_features(user_id)
        
        if not user_features:
            # Fallback: If no features exist anywhere, we can't do personalized retrieval
            # You could return popular jobs here or raise an error
            print(f"No features found for user {user_id}")
            return []

    # --- STEP 2: Retrieval (Candidates) ---
    # Passes fresh features to trigger User Tower -> ANN Search
    candidate_ids = await retrieval_client.get_candidate_ids(user_features)
    
    if not candidate_ids:
        return []

    # --- STEP 3: Ranking (Final List) ---
    # Passes candidate IDs + features for deep scoring
    ranked_results = await ranking_client.get_ranked_results(
        user_id=user_id,
        job_ids=candidate_ids,
        features=user_features
    )

    # Note: Your Ranking service returns fully hydrated job objects, 
    # so we return 'ranked_results' directly to the user.
    return ranked_results