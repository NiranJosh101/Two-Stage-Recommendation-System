# src/ingestion/users/generator.py

import uuid
import random
from datetime import datetime
from typing import List, Dict


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

    EXPERIENCE_LEVELS = ["junior", "mid", "senior"]
    EDUCATION_LEVELS = ["bachelors", "masters", "phd", "bootcamp"]
    LOCATIONS = [None, "remote", "USA", "UK", "Canada", "Germany"]

    def __init__(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)

    def _generate_single_user(self) -> Dict:
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

    def generate(self, num_users: int) -> List[Dict]:
        """
        Generate raw synthetic users.
        """
        return [self._generate_single_user() for _ in range(num_users)]
