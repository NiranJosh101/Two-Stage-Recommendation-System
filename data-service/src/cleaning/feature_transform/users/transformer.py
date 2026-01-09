from typing import List, Dict
import pandas as pd
import numpy as np

from src.cleaning.feature_transform.embeddings.embedder import get_text_embedding

class UserFeatureTransformer:
    
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embedding_model = embedding_model

    def transform_one(self, user: Dict) -> Dict:
        """
        Transform a single user dict into FS-ready features.
        """

    
        skills_text = " ".join(user.get("skills", []))
        roles_text = " ".join(user.get("primary_roles", []))
        skills_emb = get_text_embedding(skills_text, model=self.embedding_model)
        roles_emb = get_text_embedding(roles_text, model=self.embedding_model)

        
        experience_levels = ["junior", "mid", "senior", "lead", "unknown"]
        education_levels = ["high_school", "bachelor", "master", "phd", "unknown"]
        locations = ["remote", "on_site", "hybrid", "unknown"]

        experience_one_hot = {f"exp_{lvl}": int(user.get("experience_level", "unknown").lower() == lvl) 
                              for lvl in experience_levels}
        education_one_hot = {f"edu_{lvl}": int(user.get("education_level", "unknown").lower() == lvl) 
                             for lvl in education_levels}
        location_one_hot = {f"loc_{loc}": int(user.get("location", "unknown").lower() == loc) 
                            for loc in locations}

       
        years_exp = user.get("years_of_experience", np.nan)

        
        features = {
            "user_id": user["user_id"],
            "skills_emb": skills_emb,
            "roles_emb": roles_emb,
            "years_of_experience": years_exp,
            **experience_one_hot,
            **education_one_hot,
            **location_one_hot
        }

        return features

    def transform_many(self, users: List[Dict]) -> pd.DataFrame:
        """
        Transform a list of user dicts into a DataFrame ready for FS ingestion.
        """
        transformed = [self.transform_one(user) for user in users]
        return pd.DataFrame(transformed)
