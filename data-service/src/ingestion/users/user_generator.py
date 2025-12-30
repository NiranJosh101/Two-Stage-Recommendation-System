import sys
import uuid
import random
from datetime import datetime
from typing import List, Dict
from src.config.config_entities import UserDataIngestionConfig  
from src.utils.exception import RecommendationsystemDataServie
from src.utils.logging import logging


class UserGenerator:
    """
    Synthetic user source.

    """

    JOB_QUERIES = [
        "developer", "designer", "marketing",
        "sales", "data", "engineer", "product"
    ]

    ROLE_SKILLS = {
        "developer": [
            "python", "javascript", "typescript", "git",
            "django", "flask", "react", "node"
        ],
        "engineer": [
            "python", "java", "system_design",
            "docker", "kubernetes", "aws", "gcp"
        ],
        "data": [
            "sql", "python", "pandas", "numpy",
            "ml", "statistics", "data_analysis"
        ],
        "designer": [
            "figma", "ui_design", "ux_research",
            "prototyping", "visual_design"
        ],
        "marketing": [
            "seo", "content_marketing",
            "growth", "analytics", "copywriting"
        ],
        "sales": [
            "lead_generation", "crm",
            "negotiation", "pipeline_management"
        ],
        "product": [
            "product_strategy", "roadmapping",
            "stakeholder_management", "analytics"
        ]
    }

    def __init__(self, users_config: UserDataIngestionConfig, seed: int | None = None):
        try:
            if seed is not None:
                random.seed(seed)
            self.EXPERIENCE_LEVELS = users_config.experience_levels
            self.EDUCATION_LEVELS = users_config.education_levels
            self.LOCATIONS = users_config.locations
            self.num_users = users_config.num_users
        except Exception as e:
            raise RecommendationsystemDataServie(e, sys)
        

    def _generate_single_user(self) -> Dict:
        try:
            user_id = str(uuid.uuid4())

            primary_roles = random.sample(
                self.JOB_QUERIES, random.randint(1, 2)
            )

            skills = set()
            for role in primary_roles:
                role_skills = self.ROLE_SKILLS[role]
                skills.update(
                    random.sample(role_skills, random.randint(2, 4))
                )

            return {
                "user_id": user_id,
                "primary_roles": primary_roles,
                "skills": list(skills),
                "experience_level": random.choice(self.EXPERIENCE_LEVELS),
                "education_level": random.choice(self.EDUCATION_LEVELS),
                "location": random.choice(self.LOCATIONS),
                "years_of_experience": random.randint(0, 15),
                "created_at": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            raise RecommendationsystemDataServie(e, sys)
    

    def generate(self) -> List[Dict]:
        try:
            users = [self._generate_single_user() for _ in range(self.num_users)]
            logging.info("Generated %d users", self.num_users)
            return users
        except Exception as e:
            logging.exception("User Generation Failed")
            raise RecommendationsystemDataServie(e, sys)

