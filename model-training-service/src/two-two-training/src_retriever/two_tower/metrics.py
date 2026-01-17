from typing import List
import torch
import torch.nn.functional as F

def _compute_cosine_similarity(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the cosine similarity matrix.
    Shape: [num_users, num_items]
    """
    # Normalize to unit length so dot product == cosine similarity
    u_norm = F.normalize(user_embeddings, p=2, dim=1)
    i_norm = F.normalize(item_embeddings, p=2, dim=1)
    
    # Efficiently compute all-to-all similarity
    return torch.matmul(u_norm, i_norm.T)


def recall_at_k(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    true_item_indices: List[List[int]],
    k: int,
) -> float:
    """
    Recall@K: Measures what percentage of relevant items were retrieved.
    """
    scores = _compute_cosine_similarity(user_embeddings, item_embeddings)
    # top_k shape: [num_users, k]
    top_k = torch.topk(scores, k=k, dim=1).indices

    hits = 0
    total_relevant = 0

    for user_idx, relevant_items in enumerate(true_item_indices):
        if not relevant_items:
            continue
            
        retrieved = set(top_k[user_idx].tolist())
        hits += sum(1 for item in relevant_items if item in retrieved)
        total_relevant += len(relevant_items)

    return hits / max(total_relevant, 1)


def mrr_at_k(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    true_item_indices: List[List[int]],
    k: int,
) -> float:
    """
    MRR@K: Measures how high up the first relevant item appears.
    """
    scores = _compute_cosine_similarity(user_embeddings, item_embeddings)
    top_k = torch.topk(scores, k=k, dim=1).indices

    mrr_sum = 0.0

    for user_idx, relevant_items in enumerate(true_item_indices):
        relevant_set = set(relevant_items)
        for rank, item_idx in enumerate(top_k[user_idx].tolist(), start=1):
            if item_idx in relevant_set:
                mrr_sum += 1.0 / rank
                break # Only count the first hit

    return mrr_sum / max(len(true_item_indices), 1)


def ndcg_at_k(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    true_item_indices: List[List[int]],
    k: int,
) -> float:
    """
    NDCG@K: Measures retrieval quality accounting for the position of hits.
    """
    scores = _compute_cosine_similarity(user_embeddings, item_embeddings)
    top_k = torch.topk(scores, k=k, dim=1).indices
    
    # Pre-calculate log discounts for speed
    discounts = torch.log2(torch.arange(2, k + 2).float())

    ndcg_list = []

    for user_idx, relevant_items in enumerate(true_item_indices):
        if not relevant_items:
            ndcg_list.append(0.0)
            continue
            
        relevant_set = set(relevant_items)
        dcg = 0.0
        
        # Calculate DCG
        for i, item_idx in enumerate(top_k[user_idx].tolist()):
            if item_idx in relevant_set:
                dcg += 1.0 / discounts[i].item()

        # Calculate IDCG (Ideal DCG)
        num_to_sum = min(len(relevant_items), k)
        idcg = (1.0 / discounts[:num_to_sum]).sum().item()

        ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)

    return sum(ndcg_list) / max(len(ndcg_list), 1)