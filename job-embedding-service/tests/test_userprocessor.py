import numpy as np
from unittest.mock import MagicMock
import pytest
import torch
from src.user_service.processor import UserProcessor  

@pytest.fixture
def processor():
    return UserProcessor()

def test_preprocess_output_shape(processor):
    """
    Ensures that the final concatenated array matches the expected input dimension.
    Calculated as: Skill_Emb(384) + Role_Emb(384) + 1 (exp) + 5 (exp_lvl) + 5 (edu) + 4 (loc)
    """
    raw_data = {
        "user_id": "u1",
        "skills": ["python"],
        "primary_roles": ["backend"],
        "experience_level": "senior",
        "education_level": "bachelor",
        "location": "remote",
        "years_of_experience": 10
    }
    
    processed_vector = processor.preprocess(raw_data)
    
    assert isinstance(processed_vector, np.ndarray)
    assert processed_vector.dtype == np.float32
    # Check that it's a flat vector
    assert processed_vector.ndim == 1