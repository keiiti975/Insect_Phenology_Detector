import numpy as np


def get_randomsampled_idx(Y, train_idx):
    """
        randomsample with smallest idx and get idxs
        - Y <Array[int]> : Array[label]
        - train_idx <List[int]> : List[idx]
    """
    train_idx = np.asarray(train_idx)
    train_Y = Y[train_idx]
    idx, counts = np.unique(train_Y, return_counts=True)
    min_count = counts.min()
    new_train_idx = []
    for insect_id in idx:
        id_filter = train_Y == insect_id
        filtered_id = train_idx[id_filter]
        sampled_filtered_id = np.random.choice(filtered_id, min_count, replace=False)
        new_train_idx.extend(sampled_filtered_id)
    return new_train_idx