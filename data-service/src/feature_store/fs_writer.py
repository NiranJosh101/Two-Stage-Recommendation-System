from typing import Dict
import pandas as pd
from feast import FeatureStore, Entity, FeatureView, ValueType, FileSource
from pathlib import Path


from src.feature_store.fs_contract import (
    USER_FEATURE_PRIMARY_KEYS,
    JOB_FEATURE_PRIMARY_KEYS,
    TRAINING_DATASET_PRIMARY_KEYS,
)

def write_feature_group_feast(
    df: pd.DataFrame,
    feature_group_name: str,
    version: str,
    feast_repo_path: str,
    event_timestamp_column: str = None,
) -> None:
    """
    Writes a feature group to Feast (offline store) in a versioned way.

    Args:
        df: Pandas DataFrame with features.
        feature_group_name: name of the feature group.
        version: version string, e.g., 'v1'.
        feast_repo_path: path to Feast repository (repo.yaml must exist here).
        event_timestamp_column: optional, column used as event timestamp.
    """

    # Ensure Feast repo path exists
    repo_path = Path(feast_repo_path)
    if not repo_path.exists():
        raise FileNotFoundError(f"Feast repo path not found: {feast_repo_path}")

    # Initialize Feast client
    fs = FeatureStore(repo_path=str(repo_path))

    # 1️⃣ Save DataFrame as parquet to temporary location (offline source)
    temp_parquet = repo_path / f".{feature_group_name}_{version}.parquet"
    df.to_parquet(temp_parquet, index=False)

    # 2️⃣ Define FileSource for offline store
    source = FileSource(
        path=str(temp_parquet),
        event_timestamp_column=event_timestamp_column
        if event_timestamp_column else "created_at",  # fallback if needed
        created_timestamp_column=None
    )

    # 3️⃣ Define entity based on primary keys
    # Fetch primary keys from contracts
   

    pk_map: Dict[str, list] = {
        "user_features": USER_FEATURE_PRIMARY_KEYS,
        "job_features": JOB_FEATURE_PRIMARY_KEYS,
        "training_dataset": TRAINING_DATASET_PRIMARY_KEYS,
    }

    entity_keys = pk_map.get(feature_group_name)
    if entity_keys is None:
        raise ValueError(f"No primary keys defined for {feature_group_name}")

    if len(entity_keys) != 1:
        # Feast only supports 1-PK entity per FeatureView; for composite keys we handle as "composite"
        entity_name = "_".join(entity_keys)
        df[entity_name] = df[entity_keys].agg("_".join, axis=1)
        entity_keys = [entity_name]
    else:
        entity_name = entity_keys[0]

    # 4️⃣ Create Entity in Feast (if not exists)
    for pk in entity_keys:
        try:
            fs.get_entity(pk)
        except Exception:
            fs.apply([Entity(name=pk, join_keys=[pk], value_type=ValueType.STRING)])

    # 5️⃣ Define FeatureView
    feature_view_name = f"{feature_group_name}_{version}"

    feature_defs = []
    for col in df.columns:
        if col not in entity_keys:
            # All non-PK columns become features
            feature_defs.append(
                (col, ValueType.FLOAT)  # default to float, can extend with type mapping
            )

    fv = FeatureView(
        name=feature_view_name,
        entities=entity_keys,
        ttl=None,
        features=feature_defs,
        batch_source=source,
        online=False,
    )

    fs.apply([fv])

    print(f"Feature group {feature_group_name} v{version} ingested into Feast offline store at {feast_repo_path}")
