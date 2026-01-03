import pandas as pd
from typing import Dict, Any

class SchemaValidationError(Exception):
    """Raised when a dataset fails schema validation."""
    pass


def validate_schema(df: pd.DataFrame, contract: Dict[str, Any], dataset_name: str = "") -> None:
    """
    Validate a DataFrame against its schema contract.
    
    """
    fields = contract["fields"]
    errors = []

   
    for field_name, props in fields.items():
        if props.get("required", False):
            if field_name not in df.columns:
                errors.append(f"Missing required field '{field_name}' in {dataset_name}")

  
    for field_name, props in fields.items():
        if field_name not in df.columns:
            continue  
        expected_type = props["type"]
        nullable = props.get("nullable", True)

        
        if not nullable and df[field_name].isnull().any():
            errors.append(f"Field '{field_name}' contains nulls but is not nullable in {dataset_name}")

        
        non_null_series = df[field_name].dropna()
        if not non_null_series.empty:
            if expected_type == "datetime":
                if not pd.api.types.is_datetime64_any_dtype(non_null_series):
                    errors.append(f"Field '{field_name}' is not datetime in {dataset_name}")
            elif expected_type == list:
                if not all(isinstance(x, list) for x in non_null_series):
                    errors.append(f"Field '{field_name}' is not list in {dataset_name}")
            else:
                if not all(isinstance(x, expected_type) for x in non_null_series):
                    errors.append(f"Field '{field_name}' is not of type {expected_type.__name__} in {dataset_name}")

    if errors:
        raise SchemaValidationError("\n".join(errors))
