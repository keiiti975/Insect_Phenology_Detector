import numpy as np


def adopt_sampling(Y, idx, sampling):
    """
        adopt sampling to idx
        Args:
            - Y: np.array, insect labels
            - idx: [[int, ...], ...], target of sampling
            - sampling: str, choice [RandomSample, OverSample] or None
    """
    if sampling == "RandomSample":
        print("sampling = RandomSample")
        new_idx = get_randomsampled_idx(Y, idx)
    elif sampling == "OverSample":
        print("sampling == OverSample")
        new_idx = get_oversampled_idx(Y, idx)
    else:
        print("sampling = None")
        new_idx = idx
    return new_idx


def get_randomsampled_idx(Y, idx):
    """
        randomsample with smallest idx and get idxs
        Args:
            - Y: np.array, insect labels
            - idx: [[int, ...], ...], target of sampling
    """
    idx = np.asarray(idx)
    idx_filtered_Y = Y[idx]
    insect_ids, count = np.unique(idx_filtered_Y, return_counts=True)
    min_count = count.min()
    new_idx = []
    for insect_id in insect_ids:
        id_filter = idx_filtered_Y == insect_id
        filtered_idx = idx[id_filter]
        sampled_filtered_idx = np.random.choice(filtered_idx, min_count, replace=False)
        new_idx.extend(sampled_filtered_idx)
    return new_idx


def get_oversampled_idx(Y, idx):
    """
        oversample with largest idx and get idxs
        Args:
            - Y: np.array, insect labels
            - idx: [int, ...], target of sampling
    """
    idx = np.asarray(idx)
    idx_filtered_Y = Y[idx]
    insect_ids, count = np.unique(idx_filtered_Y, return_counts=True)
    max_count = count.max()
    new_idx = []
    for insect_id in insect_ids:
        id_filter = idx_filtered_Y == insect_id
        filtered_idx = idx[id_filter]
        oversampled_filtered_idx = oversample_idx_to_max_count(filtered_idx, max_count)
        new_idx.extend(oversampled_filtered_idx)
    return new_idx

    
def oversample_idx_to_max_count(filtered_idx, max_count):
    """
        oversample id up to max_count
        Args:
            - filtered_idx: [int, ...], insect_id filtered idx
            - max_count: int
    """
    now_idx_count = 0
    oversampled_idx = []
    while now_idx_count < max_count:
        if (filtered_idx.shape[0] + now_idx_count) <= max_count:
            oversampled_idx.extend(filtered_idx)
            now_idx_count += filtered_idx.shape[0]
        elif max_count < (filtered_idx.shape[0] + now_idx_count) and now_idx_count < max_count:
            random_sampled_idx = np.random.choice(filtered_idx, max_count - now_idx_count, replace=False)
            oversampled_idx.extend(random_sampled_idx)
            now_idx_count += max_count - now_idx_count
    return oversampled_idx