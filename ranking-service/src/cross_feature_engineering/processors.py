import pandas as pd
import re
from typing import List, Dict, Any

class FeatureProcessor:
    def __init__(self):
        # Must match your training config
        self.feature_cols = ["skill_overlap_score", "experience_gap"]
        self.group_col = "user_id" 

    def _get_simulated_overlap(self, job: Dict) -> float:
        # Dummy logic: real project would use NLP here
        return 0.7415 if "data" in job.get("job_title", "").lower() else 0.4500

    def _get_simulated_gap(self, job: Dict) -> int:
        # Dummy logic: real project would parse description
        return -2

    def create_grouped_dataset(self, user_id: str, hydrated_jobs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        1. Pairs the user with all candidates.
        2. Calculates cross-features.
        3. Sorts/Groups by user_id to match training data preparation.
        """
        rows = []

        for job in hydrated_jobs:
            row = {
                "user_id": user_id,
                "job_id": job.get("job_id"),
                "skill_overlap_score": self._get_simulated_overlap(job),
                "experience_gap": self._get_simulated_gap(job)
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # CRITICAL STEP: Match your training logic
        # You used GroupShuffleSplit and grouped by user_id. 
        # We sort here to ensure the DMatrix 'group' parameter stays consistent.
        df = df.sort_values(by=self.group_col).reset_index(drop=True)
        
        return df