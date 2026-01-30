import re
from datetime import datetime
from typing import Optional, List


class UserNormalizers:
    """
    Collection of static normalization utilities for user fields.

    """


    @staticmethod
    def normalize_string(value: Optional[str]) -> Optional[str]:
      
        if value is None:
            return None
        return value.strip().lower()

    @staticmethod
    def normalize_string_list(values: Optional[List[str]]) -> List[str]:
        
        if not values:
            return []

        seen = set()
        normalized = []

        for v in values:
            if v is None:
                continue

            item = str(v).strip().lower()
            if not item:
                continue

            if item not in seen:
                seen.add(item)
                normalized.append(item)

        return normalized


    @staticmethod
    def normalize_experience_level(value: Optional[str]) -> Optional[str]:
     
        return UserNormalizers.normalize_string(value)

    @staticmethod
    def normalize_education_level(value: Optional[str]) -> Optional[str]:
      
        return UserNormalizers.normalize_string(value)

    @staticmethod
    def normalize_location(value: Optional[str]) -> Optional[str]:
     
        if value is None:
            return None

        text = value.strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def normalize_years_of_experience(
        value: Optional[int | float]
    ) -> Optional[float]:
       
        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

   