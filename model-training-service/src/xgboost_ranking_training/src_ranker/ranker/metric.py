import numpy as np
from sklearn.metrics import ndcg_score

def compute_metrics(y_true, y_pred, group_ids):
    """
    Computes ranking metrics by looking at each user's 'bucket' of jobs.
    """
    # Create a temporary dataframe to make grouping easy
    import pandas as pd
    results = pd.DataFrame({
        'user_id': group_ids,
        'true_label': y_true,
        'pred_score': y_pred
    })

    ndcg_list = []

    # Calculate NDCG for every user individually
    for user_id, group in results.groupby('user_id'):
        # We need at least one positive and one negative (or two items) 
        # to calculate a meaningful ranking score
        if len(group) > 1 and group['true_label'].nunique() > 1:
            # ndcg_score expects (n_samples, n_items)
            # We treat this user's jobs as a single horizontal list
            actual = [group['true_label'].values]
            predicted = [group['pred_score'].values]
            
            score = ndcg_score(actual, predicted, k=5) # NDCG@5
            ndcg_list.append(score)

    return {
        "mean_ndcg_at_5": np.mean(ndcg_list) if ndcg_list else 0.0
    }