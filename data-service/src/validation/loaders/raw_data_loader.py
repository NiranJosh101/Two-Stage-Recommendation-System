import pandas as pd
from typing import Dict


from pathlib import Path
import pandas as pd


def _read_raw(path: str) -> pd.DataFrame:
    p = Path(path)

    
    if p.is_dir():
        dfs = []
        for file in sorted(p.iterdir()):
            if file.suffix == ".json":
                dfs.append(pd.read_json(file, lines=True))
            elif file.suffix == ".parquet":
                dfs.append(pd.read_parquet(file))

        if not dfs:
            raise ValueError(f"No supported raw files found in {path}")

        return pd.concat(dfs, ignore_index=True)

   
    if p.suffix == ".json":
        return pd.read_json(p, lines=True)

    if p.suffix == ".parquet":
        return pd.read_parquet(p)

    raise ValueError(f"Unsupported raw data format: {path}")



def load_jobs_raw(path: str) -> pd.DataFrame:
    """
    Load JobsRaw dataset.
    """
    return _read_raw(path)


def load_users_raw(path: str) -> pd.DataFrame:
    """
    Load UsersRaw dataset.
    """
    return _read_raw(path)


def load_interactions_raw(path: str) -> pd.DataFrame:
    """
    Load InteractionsRaw dataset.
    Adds a synthetic primary key 'interaction_id' for validation.
    """
    df = _read_raw(path).copy()
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
    return {
        "jobs_raw": load_jobs_raw(paths["jobs"]),
        "users_raw": load_users_raw(paths["users"]),
        "interactions_raw": load_interactions_raw(paths["interactions"]),
    }
