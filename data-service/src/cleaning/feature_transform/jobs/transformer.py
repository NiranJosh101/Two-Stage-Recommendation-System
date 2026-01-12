from typing import List, Dict
import pandas as pd
import numpy as np

from src.cleaning.feature_transform.embeddings.embedder import get_text_embedding

class JobFeatureTransformer:

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embedding_model = embedding_model

    def transform_one(self, job: Dict) -> Dict:
        """
        Transform a single job dict into FS-ready features with job_embedding.
        """
        import numpy as np

        # ---------------------------
        # Embeddings
        # ---------------------------
        title_emb = get_text_embedding(job.get("job_title", ""))
        desc_emb = get_text_embedding(job.get("job_description", ""))
        employer_emb = get_text_embedding(job.get("employer_name", ""))

       
        employment_types = ["full_time", "part_time", "contract", "internship", "unknown"]
        employment_type = job.get("job_employment_type", "unknown").lower()
        employment_one_hot = {f"employment_{t}": int(employment_type == t) for t in employment_types}

        is_remote = int(job.get("job_is_remote", False))

        salary_min = job.get("job_min_salary", 0.0)
        salary_max = job.get("job_max_salary", 0.0)
        salary_avg = 0.0
        if salary_min is not None and salary_max is not None:
            salary_avg = (salary_min + salary_max) / 2

       
        job_embedding = np.concatenate([
            title_emb,
            desc_emb,
            employer_emb,
            np.array([salary_avg], dtype=float),
            np.array([is_remote], dtype=float),
            np.array(list(employment_one_hot.values()), dtype=float)
        ])

        features = {
            "job_id": job["job_id"],
            "job_embedding": job_embedding.tolist() 
        }

        return features

    def transform_many(self, jobs: List[Dict]) -> pd.DataFrame:
        """
        Transform a list of job dicts into a DataFrame ready for FS ingestion.
        """
        transformed = [self.transform_one(job) for job in jobs]
        return pd.DataFrame(transformed)
