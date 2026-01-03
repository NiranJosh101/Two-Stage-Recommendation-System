import pandas as pd
from typing import Dict


def load_jobs_raw(path: str) -> pd.DataFrame:
    """
    Load JobsRaw dataset.
    """
    df = pd.read_parquet(path)
    return df


def load_users_raw(path: str) -> pd.DataFrame:
    """
    Load UsersRaw dataset.
    """
    df = pd.read_parquet(path)
    return df


def load_interactions_raw(path: str) -> pd.DataFrame:
    """
    Load InteractionsRaw dataset.
    Adds a synthetic primary key 'interaction_id' for validation.
    """
    df = pd.read_parquet(path)
    df = df.copy()
    df["interaction_id"] = df.index.astype(str)  
    return df


def load_all_raw_data(paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Load all raw datasets in one call.
    
    Args:
        paths: dict with keys 'jobs', 'users', 'interactions' pointing to file paths.
        
    Returns:
        dict mapping dataset names to DataFrames
    """
    data = {}
    data["jobs_raw"] = load_jobs_raw(paths["jobs"])
    data["users_raw"] = load_users_raw(paths["users"])
    data["interactions_raw"] = load_interactions_raw(paths["interactions"])
    return data
