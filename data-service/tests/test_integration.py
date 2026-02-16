import pandas as pd
import pytest
from unittest.mock import patch
from src.validation.run_validation import run_validation


@patch('src.validation.run_validation.load_jobs_raw')
@patch('src.validation.run_validation.load_users_raw')
@patch('src.validation.run_validation.load_interactions_raw')
def test_full_validation_flow_success(mock_inter, mock_users, mock_jobs):
    """
    Goal: Ensure the 'run_validation' function orchestrates all steps 
    correctly when given perfect data.
    """

    # --- 1. SETUP (The "Mocks") ---
    # We define exactly what these functions should return when called.
    # This creates a "Golden Path" where the data is 100% valid.
    mock_jobs.return_value = pd.DataFrame({
        "job_id": ["j1"], 
        "title": ["Software Engineer"]
    })
    
    mock_users.return_value = pd.DataFrame({
        "user_id": ["u1"], 
        "name": ["Alice"]
    })
    
    # Interactions must reference 'u1' and 'j1' to pass 
    # your Referential Integrity check!
    mock_inter.return_value = pd.DataFrame({
        "user_id": ["u1"], 
        "job_id": ["j1"]
    })
    
    # --- 2. ACT (The Execution) ---
    # Call the actual function you wrote in your data service.
    # It will use our 'mock_jobs', etc., instead of real loaders.
    result = run_validation()
    
    # --- 3. ASSERT (The Verification) ---
    # We verify that the function finished successfully and 
    # returned a dictionary containing our validated data.
    assert "jobs_raw" in result, "Result dictionary missing 'jobs_raw' key"
    assert "users_raw" in result, "Result dictionary missing 'users_raw' key"
    assert len(result["jobs_raw"]) == 1, "The DataFrame was not passed through correctly"
    
    # We can also verify that our mocks were actually called
    mock_jobs.assert_called_once()