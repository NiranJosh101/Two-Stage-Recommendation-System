from collections import defaultdict
from typing import Iterable, Dict, List
import os
import json

from src.supervision.labeling.label_mapper import LabelMapper, LabelingPolicy
from src.supervision.labeling.conflict_resolver import ConflictResolver, ConflictPolicy
from src.config.config_manager import ConfigurationManager

from src.utils.exception import RecommendationsystemDataServie
from src.utils.logging import logging

def generate_labeled_positives(
    interactions: Iterable[Dict[str, str]],
    policy_path: str,
) -> List[Dict[str, str]]:
    try:
        # Load policies
        labeling_policy = LabelingPolicy.from_yaml(policy_path)
        conflict_policy = ConflictPolicy.from_yaml(policy_path)

        mapper = LabelMapper(labeling_policy)
        resolver = ConflictResolver(conflict_policy)

        # Map events → labels 
        grouped_events = defaultdict(list)

        for row in interactions:
            label = mapper.map_event(row["event_type"])
            if label is None:
                continue

            key = (row["user_id"], row["job_id"])
            grouped_events[key].append((row["event_type"], label))

    
    
        # Resolve conflicts
        labeled_positives = []

        for (user_id, job_id), events in grouped_events.items():
            final_label = resolver.resolve(events)
            labeled_positives.append(
                {
                    "user_id": user_id,
                    "job_id": job_id,
                    "label": final_label,
                }
            )

        return labeled_positives
    
    except Exception as e:
        logging.error(f"Error in generating labeled positives: {e}")
        raise RecommendationsystemDataServie from e


if __name__ == "__main__":
    config = ConfigurationManager()
    interaction_ingestion_config = config.get_interaction_ingestion_config()

    INPUT_PATH = interaction_ingestion_config.interaction_clean_path
    OUTPUT_PATH = interaction_ingestion_config.interaction_labeled_positives_path
    POLICY_PATH = interaction_ingestion_config.interaction_policy_path

    # Load raw interactions from JSON
    with open(INPUT_PATH, "r") as f:
        interactions = json.load(f)

    labeled = generate_labeled_positives(
        interactions=interactions,
        policy_path=POLICY_PATH,
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Write output to JSON
    with open(OUTPUT_PATH, "w") as f:
        json.dump(labeled, f, indent=2)
    logging.info(f"✔ Wrote {len(labeled)} labeled positives → {OUTPUT_PATH}")
    print(f"✔ Wrote {len(labeled)} labeled positives → {OUTPUT_PATH}")
