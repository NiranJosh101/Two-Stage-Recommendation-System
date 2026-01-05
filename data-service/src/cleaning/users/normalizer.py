import re
from datetime import datetime
from typing import Optional, List


class UserNormalizers:
    """
    Collection of static normalization utilities for user fields.

    Stateless by design.
    Deterministic only.
    """

    # -----------------------------
    # Generic helpers
    # -----------------------------

    @staticmethod
    def normalize_string(value: Optional[str]) -> Optional[str]:
        """
        Normalize free-text string fields.
        - Lowercase
        - Strip whitespace
        - Preserve None
        """
        if value is None:
            return None
        return value.strip().lower()

    @staticmethod
    def normalize_string_list(values: Optional[List[str]]) -> List[str]:
        """
        Normalize list[str] fields.
        - None → empty list
        - Lowercase
        - Strip whitespace
        - Remove empty values
        - Deduplicate (order-preserving)
        """
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

    # -----------------------------
    # Field-specific normalizers
    # -----------------------------

    @staticmethod
    def normalize_experience_level(value: Optional[str]) -> Optional[str]:
        """
        Normalize experience_level.
        No inference, no enum enforcement.
        """
        return UserNormalizers.normalize_string(value)

    @staticmethod
    def normalize_education_level(value: Optional[str]) -> Optional[str]:
        """
        Normalize education_level.
        Sample already uses snake_case.
        """
        return UserNormalizers.normalize_string(value)

    @staticmethod
    def normalize_location(value: Optional[str]) -> Optional[str]:
        """
        Normalize location field.
        - Lowercase
        - Collapse internal whitespace
        """
        if value is None:
            return None

        text = value.strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def normalize_years_of_experience(
        value: Optional[int | float]
    ) -> Optional[float]:
        """
        Normalize years_of_experience.
        - Cast to float
        - Preserve None if invalid
        """
        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # @staticmethod
    # def normalize_created_at(value: Optional[str]) -> Optional[str]:
    #     """
    #     Normalize created_at timestamp.
    #     - Ensure ISO-8601
    #     - Preserve None if invalid
    #     """
    #     if value is None:
    #         return None

    #     try:
    #         dt = datetime.fromisoformat(value)
    #         return dt.isoformat()
    #     except ValueError:
    #         return None
