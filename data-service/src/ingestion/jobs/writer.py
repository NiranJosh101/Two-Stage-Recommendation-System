import json
from typing import List, Dict
from pathlib import Path

class JobWriter:
    def __init__(self, base_path: str = "data/bronze/jobs"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def write(self, jobs: List[Dict], filename: str) -> None:
        file_path = self.base_path / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(jobs, f, ensure_ascii=False, indent=2)
