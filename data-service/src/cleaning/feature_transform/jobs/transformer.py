from typing import List, Dict
import pandas as pd
import numpy as np

from src.cleaning.feature_transform.embeddings.embedder import get_text_embedding

class JobFeatureTransformer:
    

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embedding_model = embedding_model

    def transform_one(self, job: Dict) -> Dict:
        """
        Transform a single job dict into FS-ready features.
        """

       
        title_emb = get_text_embedding(job.get("job_title", ""), model=self.embedding_model)
        desc_emb = get_text_embedding(job.get("job_description", ""), model=self.embedding_model)
        employer_emb = get_text_embedding(job.get("employer_name", ""), model=self.embedding_model)

     
        employment_types = ["full_time", "part_time", "contract", "internship", "unknown"]
        employment_type = job.get("job_employment_type", "UNKNOWN").lower()
        employment_one_hot = {f"employment_{t}": int(employment_type == t) for t in employment_types}

     
        is_remote = int(job.get("job_is_remote", False))

        
        salary_min = job.get("job_min_salary")
        salary_max = job.get("job_max_salary")
        salary_avg = np.nan
        if salary_min is not None and salary_max is not None and not (np.isnan(salary_min) or np.isnan(salary_max)):
            salary_avg = (salary_min + salary_max) / 2

        
        features = {
            "job_id": job["job_id"],
            "title_emb": title_emb,
            "description_emb": desc_emb,
            "employer_emb": employer_emb,
            "salary_avg": salary_avg,
            "is_remote": is_remote,
            **employment_one_hot
        }

        return features

    def transform_many(self, jobs: List[Dict]) -> pd.DataFrame:
       
        transformed = [self.transform_one(job) for job in jobs]
        return pd.DataFrame(transformed)
