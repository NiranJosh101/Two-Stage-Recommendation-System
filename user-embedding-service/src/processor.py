import re
import numpy as np
import torch
from typing import List, Dict, Optional, Any, Union
from sentence_transformers import SentenceTransformer
from config.config_manager import ConfigurationManager

class UserNormalizers:
    @staticmethod
    def normalize_string(value: Optional[str]) -> Optional[str]:
        return value.strip().lower() if value else "unknown"

    @staticmethod
    def normalize_string_list(values: Optional[List[str]]) -> List[str]:
        if not values: return []
        seen = set()
        normalized = []
        for v in values:
            if v is None: continue
            item = str(v).strip().lower()
            if item and item not in seen:
                seen.add(item)
                normalized.append(item)
        return normalized

    @staticmethod
    def normalize_location(value: Optional[str]) -> str:
        if value is None: return "unknown"
        return re.sub(r"\s+", " ", value.strip().lower())

    @staticmethod
    def normalize_years_of_experience(value: Optional[int | float]) -> float:
        try:
            return float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

class UserProcessor:
    def __init__(self):
        self.config = ConfigurationManager()
        self.normalizer = UserNormalizers()
        
        # Load the Sentence Transformer once
        model_cfg = self.config.get_model_training_config()
        model_name = model_cfg.embed_model_names if hasattr(model_cfg, 'embed_model_names') else "all-MiniLM-L6-v2"
        self.embed_model = SentenceTransformer(model_name)
        
        # Training Mappings
        self.experience_levels = ["junior", "mid", "senior", "lead", "unknown"]
        self.education_levels = ["high_school", "bachelor", "master", "phd", "unknown"]
        self.locations = ["remote", "on_site", "hybrid", "unknown"]

    def get_text_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        """Your logic: optimized to return numpy for concatenation."""
        if isinstance(text, str):
            if not text:
                return np.zeros(self.embed_model.get_sentence_embedding_dimension())
            return self.embed_model.encode(text, convert_to_numpy=True)
        
        elif isinstance(text, list):
            if not text:
                return np.zeros(self.embed_model.get_sentence_embedding_dimension())
            processed_texts = [t if t else " " for t in text]
            # If it's a list, I might want to mean-pool them into one vector
            embeddings = self.embed_model.encode(processed_texts, convert_to_numpy=True)
            return np.mean(embeddings, axis=0) if embeddings.ndim > 1 else embeddings

    def preprocess(self, raw_user_data: Dict[str, Any]) -> np.ndarray:
        """The core 'Factory' that turns JSON into the Model's Input Tensor."""
        
        # Clean
        user_id = raw_user_data.get("user_id")
        skills = self.normalizer.normalize_string_list(raw_user_data.get("skills"))
        roles = self.normalizer.normalize_string_list(raw_user_data.get("primary_roles"))
        exp_lvl = self.normalizer.normalize_string(raw_user_data.get("experience_level"))
        edu_lvl = self.normalizer.normalize_string(raw_user_data.get("education_level"))
        loc = self.normalizer.normalize_location(raw_user_data.get("location"))
        years_exp = self.normalizer.normalize_years_of_experience(raw_user_data.get("years_of_experience"))

        # Transform
        skills_emb = self.get_text_embedding(" ".join(skills))
        roles_emb = self.get_text_embedding(" ".join(roles))

        # One-hot encoding
        def to_oh(val, categories):
            return [1.0 if val == cat else 0.0 for cat in categories]

        exp_vec = to_oh(exp_lvl, self.experience_levels)
        edu_vec = to_oh(edu_lvl, self.education_levels)
        loc_vec = to_oh(loc, self.locations)

        # Concatenate
        return np.concatenate([
            skills_emb,
            roles_emb,
            np.array([years_exp]),
            np.array(exp_vec),
            np.array(edu_vec),
            np.array(loc_vec)
        ]).astype(np.float32)