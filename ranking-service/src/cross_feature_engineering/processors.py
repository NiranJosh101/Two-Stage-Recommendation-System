import pandas as pd
from typing import List, Dict, Any

class FeatureProcessor:
    def __init__(self):
        
        self.feature_cols = ["skill_overlap_score", "experience_gap"]
        self.group_col = "user_id" 

    def _calculate_skill_overlap(self, user_skills: List[str], job_skills: List[str]) -> float:
        """
        Computes overlap. For a personal project with made-up data, 
        this ensures we always return a float even if lists are empty.
        """
        if not job_skills or not user_skills:
            return 0.0
        
        user_set = set(str(s).lower() for s in user_skills)
        job_set = set(str(s).lower() for s in job_skills)
        
        intersection = user_set.intersection(job_set)
        return float(len(intersection) / len(job_set))

    def _calculate_experience_gap(self, user_exp: Any, job_min_exp: Any) -> float:
        """
        Calculates user_exp - job_min_exp. 
        Cast to float to handle potential string inputs from Redis.
        """
        try:
            return float(user_exp) - float(job_min_exp)
        except (ValueError, TypeError):
            return 0.0

    def create_grouped_dataset(self, user_profile: Dict[str, Any], hydrated_jobs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        1. Pairs the user profile with all candidate jobs.
        2. Computes the actual cross-features.
        3. Sorts by user_id to maintain consistency for XGBoost DMatrix groups.
        
        Now with defensive mapping for Gateway field variations!
        """
        rows = []
        
       
        user_id = user_profile.get("user_id", "unknown_user")
        
       
        user_skills = (
            user_profile.get("skills") or 
            user_profile.get("primary_roles") or 
            user_profile.get("user_skills") or 
            []
        )
        
       
        user_years = (
            user_profile.get("years_of_experience") or 
            user_profile.get("experience") or 
            user_profile.get("experience_years") or 
            0
        )

        for job in hydrated_jobs:
            job_id = job.get("job_id")
            job_skills = job.get("required_skills", [])
            job_min_exp = job.get("min_experience", 0)

            rows.append({
                "user_id": user_id,
                "job_id": job_id,
                "skill_overlap_score": self._calculate_skill_overlap(user_skills, job_skills),
                "experience_gap": self._calculate_experience_gap(user_years, job_min_exp)
            })

        df = pd.DataFrame(rows)

        df = df.sort_values(by=self.group_col).reset_index(drop=True)
        
        return df