from dataclasses import dataclass
from typing import Iterable
import yaml


@dataclass(frozen=True)
class ConflictPolicy:
    priority: dict[str, int]

    @classmethod
    def from_yaml(cls, path: str) -> "ConflictPolicy":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        return cls(priority=raw["priority"])


class ConflictResolver:
    """
    Resolves label conflicts using highest-priority-wins strategy.
    """

    def __init__(self, policy: ConflictPolicy):
        self.priority = policy.priority

    def resolve(self, events: Iterable[tuple[str, int]]) -> int:
        """
        Resolve multiple events into a single label.

        """
        events = list(events)
        if not events:
            raise ValueError("No events provided for conflict resolution")

        # Select event with highest priority
        winning_event = max(
            events,
            key=lambda e: self.priority.get(e[0], -1)
        )

        return winning_event[1]



if __name__ == "__main__":
    policy = ConflictPolicy.from_yaml(
        "src/supervision/policy/labeling_policy.yaml"
    )
    resolver = ConflictResolver(policy)

    sample_events = [
        ("view", 0),
        ("click", 1),
        ("apply", 1),
    ]

    print(resolver.resolve(sample_events))  # → 1
