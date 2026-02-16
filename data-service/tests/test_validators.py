import pytest
import pandas as pd
from src.validation.validators.primary_key_validator import validate_primary_key, PrimaryKeyValidationError
from src.validation.validators.schema_validator import validate_schema, SchemaValidationError

def test_primary_key_validator_throws_error_on_duplicates():
    # create dummy data with duplicate job_ids
    df = pd.DataFrame({
        "job_id": ["job_1", "job_1"], 
        "title": ["Dev", "Dev"]
    })
    
    # verify that the custom exception is raised
    with pytest.raises(PrimaryKeyValidationError) as excinfo:
        validate_primary_key(df, primary_key="job_id", dataset_name="test_set")
    
    assert "Duplicate values found" in str(excinfo.value)

def test_schema_validator_catches_wrong_type():
    contract = {
        "fields": {
            "age": {"type": int, "required": True}
        }
    }
    # pass a string where an int is expected
    df = pd.DataFrame({"age": ["twenty"]})
    
    with pytest.raises(SchemaValidationError):
        validate_schema(df, contract, "test_set")