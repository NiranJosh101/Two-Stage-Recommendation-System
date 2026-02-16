import pytest
import numpy as np
from unittest.mock import MagicMock
from src.inference.model_server import ModelServer

def test_scoring_output_range():
    """
    Unit Test: Ensures the XGBoost wrapper returns a valid probability.
    """
    # Mocking the XGBoost model to avoid loading a large .json/.bin file
    mock_model = MagicMock()
    # Simulate a probability output (binary:logistic)
    mock_model.predict.return_value = np.array([0.85], dtype=np.float32)
    
    service = ModelServer(model=mock_model)
    
    # Dummy joined feature vector (e.g., 50 features)
    dummy_features = np.random.rand(1, 50) 
    score = service.score_candidate(dummy_features)
    
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0, "Score must be a probability between 0 and 1"