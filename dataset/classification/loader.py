import numpy as np


def create_validation_split(Y, test_ratio):
    """
        create train and test validation idx
        Args:
            - Y: np.array, insect labels
            - test_ratio: float
    """
    idx, counts = np.unique(Y, return_counts=True)
    ntests = counts * test_ratio
    ntests = np.round(ntests).astype("int")

    valid_num = int(1.0 / test_ratio)

    valid_idx = [np.where(Y == i)[0] for i in range(len(ntests))]
    default_idx = [np.where(Y == i)[0] for i in range(len(ntests))]

    test = []
    train = []
    for i in range(valid_num):
        if i == valid_num - 1:
            ntests = counts - ntests * (valid_num - 1)

        valid_test = []
        valid_train = []
        for i, n in enumerate(ntests):
            test_id = np.asarray(valid_idx[i][:n])
            train_id = list(set(default_idx[i]) - set(test_id))
            valid_test.extend(test_id.tolist())
            valid_train.extend(train_id)
            valid_idx[i] = list(set(valid_idx[i]) - set(test_id))
        test.append(valid_test)
        train.append(valid_train)

    return train, test


def load_validation_data(X, Y, valid_train_idx, valid_test_idx):
    """
        load validation data from valid_idx
        Args:
            - X: np.array, insect images
            - Y: np.array, insect labels
            - valid_train_idx: [int, ...]
            - valid_test_idx: [int, ...]
    """
    xtr = X[valid_train_idx]
    ytr = Y[valid_train_idx]
    xte = X[valid_test_idx]
    yte = Y[valid_test_idx]
    return xtr, ytr, xte, yte