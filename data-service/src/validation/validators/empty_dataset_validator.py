import pandas as pd

class EmptyDatasetError(Exception):
    """Raised when a dataset is empty."""
    pass


def validate_non_empty(df: pd.DataFrame, dataset_name: str = "") -> None:
    """
    Validate that the DataFrame is not empty.

    Raises:
        EmptyDatasetError if DataFrame is empty
    """
    if df.empty:
        raise EmptyDatasetError(f"Dataset '{dataset_name}' is empty!")
