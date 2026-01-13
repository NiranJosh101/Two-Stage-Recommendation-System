import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import timedelta

from feast import FeatureStore, Entity, FeatureView, Field, FileSource
from feast.types import Float32, String, Int64

from src.feature_store.fs_contract import (
    USER_FEATURE_PRIMARY_KEYS,
    JOB_FEATURE_PRIMARY_KEYS,
    TRAINING_DATASET_PRIMARY_KEYS,
    RANKING_FEATURE_PRIMARY_KEYS,
)

def write_feature_group_feast(
    df: pd.DataFrame,
    feature_group_name: str,
    version: str,
    feast_repo_path: str,
    event_timestamp_column: str = "event_timestamp",
) -> None:
    """
    Writes a feature group to Feast (offline store) in a versioned way.
    
    """

    repo_path = Path(feast_repo_path)
    if not repo_path.exists():
        raise FileNotFoundError(f"Feast repo path not found: {feast_repo_path}")


    fs = FeatureStore(repo_path=str(repo_path))


    pk_map: Dict[str, List[str]] = {
        "user_features": USER_FEATURE_PRIMARY_KEYS,
        "job_features": JOB_FEATURE_PRIMARY_KEYS,
        "training_dataset": TRAINING_DATASET_PRIMARY_KEYS,
        "ranking_features": RANKING_FEATURE_PRIMARY_KEYS,
    }

    entity_keys = pk_map.get(feature_group_name)
    if entity_keys is None:
        raise ValueError(f"No primary keys defined for {feature_group_name}")


    if len(entity_keys) > 1:
        entity_name = "_".join(entity_keys)
        df[entity_name] = df[entity_keys].astype(str).agg("_".join, axis=1)
        entity_keys = [entity_name]
    else:
        entity_name = entity_keys[0]


    if event_timestamp_column not in df.columns:

        df[event_timestamp_column] = pd.Timestamp.now()

   
    data_path = repo_path / "data"
    data_path.mkdir(exist_ok=True)
    
    parquet_filename = f"{feature_group_name}_{version}.parquet"
    target_parquet = data_path / parquet_filename
    df.to_parquet(target_parquet, index=False)

    source = FileSource(
        path=str(target_parquet),
        timestamp_field=event_timestamp_column,
    )


    try:
        
        entity_obj = fs.get_entity(entity_name)
    except Exception:

        entity_obj = Entity(
            name=entity_name,
            join_keys=[entity_name],
            description=f"Primary key for {feature_group_name}",
        )
        fs.apply([entity_obj])

        entity_obj = fs.get_entity(entity_name)

    feature_view_name = f"{feature_group_name}_{version}"

    schema = [
        Field(name=col, dtype=Float32) 
        for col in df.columns 
        if col not in entity_keys and col != event_timestamp_column
    ]

    fv = FeatureView(
        name=feature_view_name,
        entities=[entity_obj],  
        ttl=timedelta(days=365),
        schema=schema,
        source=source,
        online=True,
    )

    fs.apply([fv])

    print(
        f"✔✔ Feature group '{feature_group_name}' "
        f"(version={version}) successfully registered in Feast registry."
    )