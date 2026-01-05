from typing import Dict, Any
from src.cleaning.interactions.normalizer import InteractionNormalizers


class InteractionCleaner:
    """
    Applies deterministic cleaning to interactions that have already passed validation.
    Input: validated interaction dict (INTERACTIONS_RAW_CONTRACT)
    Output: cleaned interaction dict (same schema, join-safe)
    """

    def __init__(self):
        self.normalizer = InteractionNormalizers()

    def clean(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean a single interaction record.
        """
        return {
            # Primary key
            "interaction_id": interaction["interaction_id"],

            # Foreign keys (pass-through)
            "user_id": interaction["user_id"],
            "job_id": interaction["job_id"],

            # Event data
            "event_type": self.normalizer.normalize_event_type(
                interaction.get("event_type")
            ),

            # # Metadata
            # "timestamp": self.normalizer.normalize_timestamp(
            #     interaction.get("timestamp")
            # ),
        }

    def clean_many(self, interactions):
        """
        Clean multiple interaction records from a DataFrame.
        """
        interactions_list = interactions.to_dict(orient="records")
        return [self.clean(interaction) for interaction in interactions_list]
