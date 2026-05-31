from unittest.mock import MagicMock
import torch
from src.user_service.embedder import UserEmbedder
import numpy as np

def test_user_embedder_returns_normalized_vector():
    # 1. Mock the model so we don't need to load a real 500MB file
    mock_model = MagicMock()
    # Simulate the model outputting a non-normalized vector [3.0, 4.0]
    # In math, a 3-4-5 triangle means the normalized length should be 1.0
    mock_model.user_tower.return_value = torch.tensor([[3.0, 4.0]])
    mock_model.to.return_value = mock_model # Handle .to(device) calls
    
    embedder = UserEmbedder(model=mock_model, device="cpu")
    
    # 2. Input a dummy numpy array
    dummy_input = np.array([1.0, 2.0], dtype=np.float32)
    vector = embedder.compute(dummy_input)
    
    # 3. Assert: The output must be normalized (magnitude = 1.0)
    magnitude = np.linalg.norm(vector)
    assert np.isclose(magnitude, 1.0), "The embedder failed to normalize the output vector!"
    assert vector.ndim == 1 # compute() uses .flatten()
    