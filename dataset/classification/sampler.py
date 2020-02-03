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


def get_randomoversampled_idx(Y, train_idx):
    """
        randomoversample with largest idx and get idxs
        - Y <Array[int]> : Array[label]
        - train_idx <List[int]> : List[idx]
    """
    train_idx = np.asarray(train_idx)
    train_Y = Y[train_idx]
    idx, counts = np.unique(train_Y, return_counts=True)
    max_count = counts.max()
    new_train_idx = []
    for insect_id in idx:
        id_filter = train_Y == insect_id
        filtered_id = train_idx[id_filter]
        oversampled_filtered_id = get_oversampled_id(filtered_id, max_count)
        new_train_idx.extend(oversampled_filtered_id)
    return new_train_idx

    
def get_oversampled_id(filtered_id, max_count):
    """
        oversampling id up to max_count
        - filtered_id <List[int]> : List[idx]
        - max_count <int>
    """
    now_id_count = 0
    oversampled_id = []
    while now_id_count < max_count:
        if (filtered_id.shape[0] + now_id_count) <= max_count:
            oversampled_id.extend(filtered_id)
            now_id_count += filtered_id.shape[0]
        elif max_count < (filtered_id.shape[0] + now_id_count) and now_id_count < max_count:
            random_sampled_id = np.random.choice(filtered_id, max_count - now_id_count, replace=False)
            oversampled_id.extend(random_sampled_id)
            now_id_count += max_count - now_id_count
    return oversampled_id