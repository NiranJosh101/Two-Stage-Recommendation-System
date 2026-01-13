import os
import yaml
# from src.datascience import logger
import json
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any



    

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns
    
    """
    
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
        # logger.info(f"yaml file: {path_to_yaml} loaded successfully")

        ## to better read key-value pairs
        return ConfigBox(content)
    



def safe_write_json(df, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", indent=2)


import json
import pandas as pd
from typing import List, Dict


def load_clean_data(path: str) -> List[Dict]:
    """
    Load cleaned data from disk and return as list of dicts.

    Supported formats:
    - .json   (array of objects OR JSONL)
    - .csv
    - .parquet
    """

    if path.endswith(".json"):
        # Try standard JSON array first
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON file must contain a list of objects")
            return data

        # Fallback: JSON Lines (one JSON object per line)
        except json.JSONDecodeError:
            records = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            return records

    elif path.endswith(".csv"):
        df = pd.read_csv(path)
        return df.to_dict(orient="records")

    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")

    else:
        raise ValueError(f"Unsupported file format: {path}")





def write_jsonl(df, output_path: str):
    """
    Write DataFrame to newline-delimited JSON (JSONL).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record) + "\n")



def load_json(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)




def write_json(data, path: str):
    path = Path(path)  # ← convert string to Path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_json_to_df(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path_obj}")

    with open(path_obj, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)  # rewind file

        if first_char == "[":  # standard JSON array
            return pd.read_json(f)
        elif first_char == "{":  # NDJSON (one JSON object per line)
            return pd.read_json(f, lines=True)
        else:
            raise ValueError(f"Unknown JSON format in file: {path_obj}")



def flatten_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten user_features and job_features columns if they contain embeddings.
    """
    if "user_features" in df.columns:
        if isinstance(df["user_features"].iloc[0], dict) and "user_embedding" in df["user_features"].iloc[0]:
            df["user_embedding"] = df["user_features"].apply(lambda x: x["user_embedding"])
        df = df.drop(columns=["user_features"])

    if "job_features" in df.columns:
        if isinstance(df["job_features"].iloc[0], dict) and "job_embedding" in df["job_features"].iloc[0]:
            df["job_embedding"] = df["job_features"].apply(lambda x: x["job_embedding"])
        df = df.drop(columns=["job_features"])

    return df