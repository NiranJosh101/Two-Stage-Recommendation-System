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
    
    Args:
        path_to_yaml (str): path like input
    
    Raises:
        ValueError: if yaml file is empty
        e: empty files
    
    Returns:
        ConfigBox: Configbox type
    """
    
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
        # logger.info(f"yaml file: {path_to_yaml} loaded successfully")

        ## to better read key-value pairs
        return ConfigBox(content)
    
    