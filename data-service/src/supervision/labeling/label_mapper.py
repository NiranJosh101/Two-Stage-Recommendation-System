from dataclasses import dataclass
from typing import Optional
import yaml
from src.config.config_manager import ConfigurationManager


@dataclass(frozen=True)
class LabelingPolicy:
    version: str
    label_map: dict[str, int]
    ignored_events: set[str]

    @classmethod
    def from_yaml(cls, path: str) -> "LabelingPolicy":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        return cls(
            version=raw["version"],
            label_map=raw["label_map"],
            ignored_events=set(raw.get("ignored_events", [])),
        )


class LabelMapper:
    """
    Maps individual interaction events to labels using a labeling policy.
    """

    def __init__(self, policy: LabelingPolicy):
        self.policy = policy

    def map_event(self, event_type: str) -> Optional[int]:
        """
        Map a single event_type to a label.

        """
        if event_type in self.policy.ignored_events:
            return None

        if event_type not in self.policy.label_map:
            return None

        return self.policy.label_map[event_type]



if __name__ == "__main__":
    policy = LabelingPolicy.from_yaml(
        "src/supervision/policy/labeling_policy.yaml"
    )
    mapper = LabelMapper(policy)

    for evt in ["view", "click", "apply", "bookmark", "unknown"]:
        print(evt, "→", mapper.map_event(evt))
