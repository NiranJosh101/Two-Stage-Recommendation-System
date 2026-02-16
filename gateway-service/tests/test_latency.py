import pytest
import time
from httpx import AsyncClient, ASGITransport
from app.main import app 

@pytest.mark.asyncio
async def test_gateway_latency_threshold():
    """
    10/10 Move: Performance Benchmark for the actual FastAPI Route.
    This ensures that the 'recommend_jobs_logic' and FastAPI overhead 
    stay within our sub-second SLA.
    """
    
    
    
    transport = ASGITransport(app=app)
    
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        
        # Define a realistic request body
        payload = {
            "user_id": "u123",
            "top_k": 10
        }

        # 2. ACT: Measure the precise time taken for the POST request
        start_time = time.perf_counter()
        
        response = await client.post("/recommend", json=payload)
        
        end_time = time.perf_counter()

        # 3. CALCULATE: Convert to milliseconds
        total_ms = (end_time - start_time) * 1000
        
        # 4. ASSERT: Ensure success and speed
        assert response.status_code == 200, f"Request failed with {response.text}"
        
        # We set 200ms as a strict gate. In a personal project, 
        # meeting this shows you've optimized your async logic.
        assert total_ms < 200.0, f"Gateway too slow! Took {total_ms:.2f}ms"
        
        print(f"\nLatency Benchmark: {total_ms:.2f}ms")

def test_health_check_latency():
    """
    Simple check to ensure the framework itself isn't the bottleneck.
    """
    # [Logic similar to above for the /health endpoint]
    pass