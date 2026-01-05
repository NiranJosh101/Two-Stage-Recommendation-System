import re
from typing import Optional, Tuple


class JobNormalizers:
    """
    Collection of static normalization utilities for job fields.
    
    This class is stateless by design.
    It exists purely as a namespacing mechanism.
    """

   
    @staticmethod
    def normalize_string(value: Optional[str]) -> Optional[str]:
        """
        Normalize free-text string fields.
        - Lowercase
        - Strip leading/trailing whitespace
        - Preserve None
        """
        if value is None:
            return None
        return value.strip().lower()

    _EMPLOYMENT_TYPE_MAP = {
        "full-time": "FULL_TIME",
        "full time": "FULL_TIME",
        "part-time": "PART_TIME",
        "part time": "PART_TIME",
        "contract": "CONTRACT",
        "intern": "INTERN",
        "internship": "INTERN",
        "temporary": "TEMP",
        "temp": "TEMP",
    }

    @staticmethod
    def normalize_employment_type(value: Optional[str]) -> str:
        """
        Normalize employment type into a closed set.
        Unknown or missing values become 'UNKNOWN'.
        """
        if value is None:
            return "UNKNOWN"

        key = value.strip().lower()
        return JobNormalizers._EMPLOYMENT_TYPE_MAP.get(key, "UNKNOWN")

   
    @staticmethod
    def normalize_boolean(value: Optional[bool], default: bool = False) -> bool:
        """
        Normalize boolean fields.
        - If None, return default
        - Otherwise cast to bool
        """
        if value is None:
            return default
        return bool(value)


    @staticmethod
    def normalize_salary(value: Optional[float]) -> Optional[float]:
        """
        Normalize salary fields.
        - Preserve None
        - Ensure float
        """
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def normalize_salary_range(
        min_salary: Optional[float],
        max_salary: Optional[float],
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Normalize salary min/max together.
        - Ensure both are floats or None
        - Swap if min > max
        """
        min_val = JobNormalizers.normalize_salary(min_salary)
        max_val = JobNormalizers.normalize_salary(max_salary)

        if min_val is not None and max_val is not None:
            if min_val > max_val:
                return max_val, min_val

        return min_val, max_val
    

    @staticmethod
    def normalize_job_description(value: str | None) -> str | None:
        """
        Make job_description storage-safe and model-safe.
        No NLP, no semantics.
        """
        if value is None:
            return None

        text = str(value)
        
        text = re.sub(r"<[^>]+>", " ", text)

        text = re.sub(r"\s+", " ", text)

        return text.strip()
