from typing import Optional
from datetime import datetime


class InteractionNormalizers:
    """
    Collection of static normalization utilities for interaction fields.

    Stateless.
    Deterministic.
    """

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
    def normalize_event_type(value: Optional[str]) -> Optional[str]:
        """
        Normalize event_type.
        No inference, no enum enforcement.
        """
        return InteractionNormalizers.normalize_string(value)

    # @staticmethod
    # def normalize_timestamp(value: Optional[str]) -> Optional[str]:
    #     """
    #     Normalize timestamp.
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
