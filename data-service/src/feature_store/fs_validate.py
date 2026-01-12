from typing import Dict, List
import pandas as pd


def validate_schema(
    df: pd.DataFrame,
    expected_schema: Dict[str, str],
) -> None:
    """
    Enforces exact column match.
    """

    expected_columns = set(expected_schema.keys())
    actual_columns = set(df.columns)

    if actual_columns != expected_columns:
        missing = expected_columns - actual_columns
        extra = actual_columns - expected_columns

        raise ValueError(
            f"Schema mismatch. "
            f"Missing columns: {missing}. "
            f"Extra columns: {extra}."
        )


def validate_primary_keys(
    df: pd.DataFrame,
    primary_keys: List[str],
) -> None:
    """
    Enforces:
    - No null PKs
    - No duplicate PKs
    """

    # null check
    if df[primary_keys].isnull().any().any():
        raise ValueError("Null values found in primary keys")

    # duplicate check
    if df.duplicated(subset=primary_keys).any():
        raise ValueError("Duplicate primary keys detected")


def validate_feature_group(
    df: pd.DataFrame,
    schema: Dict[str, str],
    primary_keys: List[str],
) -> None:
    """
    Single validation entrypoint.
    Called before every feature store write.
    """

    validate_schema(df, schema)
    validate_primary_keys(df, primary_keys)
