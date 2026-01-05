from typing import Dict, Any
from src.cleaning.users.normalizer import UserNormalizers


class UserCleaner:
    """
    Applies deterministic cleaning to users that have already passed validation.
    Input: validated user dict (USERS_RAW_CONTRACT)
    Output: cleaned user dict (same schema, join-safe)
    """

    def __init__(self):
        self.normalizer = UserNormalizers()

    def clean(self, user: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean a single user record.
        """
        return {
            # Primary key (pass-through)
            "user_id": user["user_id"],

            # Lists
            "primary_roles": self.normalizer.normalize_string_list(
                user.get("primary_roles")
            ),
            "skills": self.normalizer.normalize_string_list(
                user.get("skills")
            ),

            # Scalars
            "experience_level": self.normalizer.normalize_experience_level(
                user.get("experience_level")
            ),
            "education_level": self.normalizer.normalize_education_level(
                user.get("education_level")
            ),
            "location": self.normalizer.normalize_location(
                user.get("location")
            ),
            "years_of_experience": self.normalizer.normalize_years_of_experience(
                user.get("years_of_experience")
            ),

            # # Metadata
            # "created_at": self.normalizer.normalize_created_at(
            #     user.get("created_at")
            # ),
        }

    def clean_many(self, users):
        """
        Clean multiple user records from a DataFrame.
        """
        users_list = users.to_dict(orient="records")
        return [self.clean(user) for user in users_list]
