import pandas as pd
import xgboost as xgb

def prepare_ranking_data(df, target_col, group_col, feature_cols):
    """
    Prepares data for XGBoost Ranking.
    Groups must be sorted by the group_col (e.g., user_id or query_id).
    """
    # XGBoost ranking requires data to be sorted by group
    df = df.sort_values(by=group_col)
    
    # Calculate the size of each group
    groups = df.groupby(group_col).size().to_frame('size')['size'].to_list()
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Create DMatrix with group information
    dtrain = xgb.DMatrix(X, label=y)
    dtrain.set_group(groups)
    
    return dtrain, X, y