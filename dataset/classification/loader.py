import h5py
import numpy as np
import os
import torch
from dataset.classification.augment import adopt_rotate, adopt_rotate_DCL, adopt_random_size_crop
from dataset.classification.region_confusion_mechanism import region_confusion_mechanism


def create_split(X, Y, test_ratio=0.2):
    idx, counts = np.unique(Y, return_counts=True)
    ntests = counts * test_ratio
    ntests = np.round(ntests).astype("int")

    test = []
    train = []
    for i, n in enumerate(ntests):
        idx = np.where(Y == i)[0]
        test_id = np.random.choice(idx, n, replace=False)
        train_id = list(set(idx) - set(test_id))
        test.extend(test_id.tolist())
        train.extend(train_id)

    assert len(test) == sum(ntests)
    assert len(train) == len(X) - len(test)

    return X[train], Y[train], X[test], Y[test]


def create_validation_split(Y, test_ratio):
    """
        create train and test validation idx
        - Y: insect labels
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


def create_dataset_from_all_data(all_data_path, train_data_path, test_data_path, test_ratio):
    if os.path.exists(test_data_path) is False:
        with h5py.File(all_data_path) as f:
            X = f["X"][:]
            Y = f["Y"][:]
        xtr, ytr, xte, yte = create_split(X, Y, test_ratio)

        with h5py.File(train_data_path) as f:
            f.create_dataset("X", data=xtr)
            f.create_dataset("Y", data=ytr)
        with h5py.File(test_data_path) as f:
            f.create_dataset("X", data=xte)
            f.create_dataset("Y", data=yte)
            

def create_train_data(xtr, ytr, rotate, augment):
    """
        adopt data augment to xtr, ytr
        - xtr: train insect images
        - ytr: train insect labels
        - rotate: float
        - argment: choice [None, "RandomSizeCrop", "RegionConfusionMechanism", "autoaugment"]
    """
    if augment == None:
        print("augment = None")
        xtr = torch.from_numpy(xtr).transpose(1, -1).float()
        ytr = torch.from_numpy(ytr)
        return xtr, ytr
    elif augment == "autoaugment":
        print("augment = autoaugment")
        return xtr, ytr
    else:
        for elem_augment in augment:
            print("augment = " + elem_augment)
            if elem_augment == "RandomSizeCrop":
                xtr = adopt_random_size_crop(xtr)
            elif elem_augment == "RegionConfusionMechanism":
                new_xtr, new_coordinate = region_confusion_mechanism(xtr)
                xtr = np.concatenate([xtr, new_xtr])
                ytr = np.concatenate([ytr, ytr])
        print("making rotate" + str(rotate) + " dataset")
        xtr, ytr = adopt_rotate(xtr, ytr, rotate)
        xtr = torch.from_numpy(xtr).transpose(1, -1).float()
        ytr = torch.from_numpy(ytr)
        return xtr, ytr


def create_train_data_DCL(xtr, ytr, target_dest_or_not, target_coordinate, rotate, augment):
    """
        adopt data augment to xtr, ytr using DCL
        - xtr <Array[int, int, int, int]> : Array[image_num, width, height, channels]
        - ytr <Array[int]> : Array[image_num]
        - target_dest_or_not <>
        - target_coordinate <>
        - rotate <>
    """
    print("making rotate" + str(rotate) + " dataset")
    if argment == "RandomSizeCrop":
        print("adopt RandomSizeCrop")
        xtr = adopt_random_size_crop(xtr)
    xtr, ytr, target_dest_or_not, target_coordinate = adopt_rotate_DCL(
        xtr, ytr, target_dest_or_not, target_coordinate, rotate)
    xtr = torch.from_numpy(xtr).transpose(1, -1).float()
    ytr = torch.from_numpy(ytr)
    target_dest_or_not = torch.from_numpy(target_dest_or_not).long()
    target_coordinate = torch.from_numpy(target_coordinate).float()
    return xtr, ytr, target_dest_or_not, target_coordinate


def load_data(train_data_path, test_data_path):
    print("loading dataset")
    with h5py.File(train_data_path) as f:
        xtr = f["X"][:]
        ytr = f["Y"][:]
    with h5py.File(test_data_path) as f:
        xte = f["X"][:]
        yte = f["Y"][:]

    _, ntests = np.unique(yte, return_counts=True)
    xte, yte = torch.from_numpy(xte).transpose(
        1, -1).float().cuda(), torch.from_numpy(yte).cuda()
    return xtr, ytr, xte, yte, ntests


def load_validation_data(X, Y, train_idx, test_idx):
    """
        load validation data from idx
        - X: insect images
        - Y: insect labels
        - train_idx: [valid_num, valid_data_size]
        - test_idx: [valid_num, valid_data_size]
    """
    xtr = X[train_idx]
    ytr = Y[train_idx]
    xte = X[test_idx]
    yte = Y[test_idx]
    xte, yte = torch.from_numpy(xte).transpose(
        1, -1).float().cuda(), torch.from_numpy(yte).cuda()
    return xtr, ytr, xte, yte


def load_semantic_vectors(semantic_save_path):
    semantic_vectors = []
    with open(semantic_save_path, mode="r") as f:
        lines = f.readlines()
        for line in lines:
            semantic_vectors.append(line.split("\n")[0].split(" "))
    semantic_vectors = np.asarray(semantic_vectors).astype("float32")
    return torch.from_numpy(semantic_vectors)
