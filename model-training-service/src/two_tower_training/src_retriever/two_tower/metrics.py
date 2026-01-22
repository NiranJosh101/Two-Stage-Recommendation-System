from typing import List
import torch
import torch.nn.functional as F
import numpy as np

def _compute_cosine_similarity(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the cosine similarity matrix.
    Shape: [num_users, num_items]
    """
    u_norm = F.normalize(user_embeddings, p=2, dim=1)
    i_norm = F.normalize(item_embeddings, p=2, dim=1)
    return torch.matmul(u_norm, i_norm.T)


def _get_relevant_set(relevant_items):
    """
    Helper to safely convert potential Tensors into a Python set of indices.
    """
    if torch.is_tensor(relevant_items):
        if relevant_items.numel() == 0:
            return set()
        return set(relevant_items.cpu().flatten().tolist())
    if not relevant_items:
        return set()
    return set(relevant_items)


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
    actual_k = min(k, item_embeddings.size(0))
    top_k = torch.topk(scores, k=actual_k, dim=1).indices

    hits = 0
    total_relevant = 0

    for user_idx, relevant_items in enumerate(true_item_indices):
        relevant_set = _get_relevant_set(relevant_items)
        if not relevant_set:
            continue
            
        retrieved = set(top_k[user_idx].cpu().tolist())
        hits += sum(1 for item in relevant_set if item in retrieved)
        total_relevant += len(relevant_set)

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
    actual_k = min(k, item_embeddings.size(0))
    top_k = torch.topk(scores, k=actual_k, dim=1).indices

    mrr_sum = 0.0

    for user_idx, relevant_items in enumerate(true_item_indices):
        relevant_set = _get_relevant_set(relevant_items)
        if not relevant_set:
            continue

        for rank, item_idx in enumerate(top_k[user_idx].cpu().tolist(), start=1):
            if item_idx in relevant_set:
                mrr_sum += 1.0 / rank
                break 

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
    actual_k = min(k, item_embeddings.size(0))
    top_k = torch.topk(scores, k=actual_k, dim=1).indices
    
    discounts = torch.log2(torch.arange(2, actual_k + 2).float()).to(scores.device)

    ndcg_list = []

    for user_idx, relevant_items in enumerate(true_item_indices):
        relevant_set = _get_relevant_set(relevant_items)
        if not relevant_set:
            ndcg_list.append(0.0)
            continue
            
        dcg = 0.0
        # Calculate DCG
        for i, item_idx in enumerate(top_k[user_idx].cpu().tolist()):
            if item_idx in relevant_set:
                dcg += 1.0 / discounts[i].item()

        # Calculate IDCG
        num_to_sum = min(len(relevant_set), actual_k)
        idcg = (1.0 / discounts[:num_to_sum]).sum().item()

        ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)

    return sum(ndcg_list) / max(len(ndcg_list), 1)