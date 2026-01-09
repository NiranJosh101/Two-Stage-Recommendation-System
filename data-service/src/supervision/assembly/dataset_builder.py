from typing import List, Dict
import random


def build_contrastive_dataset(
    labeled_positives: List[Dict[str, str]],
    sampled_negatives: List[Dict[str, str]],
    seed: int = 42,
    shuffle: bool = True,
) -> List[Dict[str, str]]:
    """
    Combine positives and negatives into a final dataset.

    """

    # Basic validation
    for row in labeled_positives:
        assert row['label'] == 1, "Positive sample has non-1 label"

    for row in sampled_negatives:
        assert row['label'] == 0, "Negative sample has non-0 label"

    dataset = labeled_positives + sampled_negatives

    if shuffle:
        random.seed(seed)
        random.shuffle(dataset)

    return dataset
