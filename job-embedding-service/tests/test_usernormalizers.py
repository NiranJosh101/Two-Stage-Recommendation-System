import pytest
import numpy as np
from src.user_service.processor import UserNormalizers

def test_normalize_string_list_removes_duplicates_and_whitespace():
    # Setup
    input_list = [" Python ", "python", "AWS", None, "  aws  "]
    # Act
    result = UserNormalizers.normalize_string_list(input_list)
    # Assert
    assert result == ["python", "aws"] # Should be lowercase, stripped, and unique

def test_normalize_years_exp_handles_bad_input():
    assert UserNormalizers.normalize_years_of_experience("5") == 5.0
    assert UserNormalizers.normalize_years_of_experience(None) == 0.0
    assert UserNormalizers.normalize_years_of_experience("abc") == 0.0