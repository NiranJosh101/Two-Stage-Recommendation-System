import pandas as pd

class PrimaryKeyValidationError(Exception):
    """Raised when primary key validation fails."""
    pass


def validate_primary_key(df: pd.DataFrame, primary_key: str, dataset_name: str = "") -> None:
    
    if primary_key not in df.columns:
        raise PrimaryKeyValidationError(
            f"Primary key '{primary_key}' not found in {dataset_name}"
        )

    duplicates = df[df.duplicated(subset=[primary_key], keep=False)]
    if not duplicates.empty:
        raise PrimaryKeyValidationError(
            f"Duplicate values found in primary key '{primary_key}' of {dataset_name}. "
            f"Duplicate rows:\n{duplicates}"
        )
