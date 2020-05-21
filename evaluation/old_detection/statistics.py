import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join as pj
import pandas as pd


def get_size(bbox): return (
    bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])


def compute_each_size_df(gt_dict):
    """
        compute each insect size & accuracy dataframe
        - gt_dict: {file id: {'bbox', 'difficult', 'det', 'default_name'}}
    """
    dets, sizes = [], []
    for im_id, gt in gt_dict.items():
        dets.append(gt["det"])
        sizes.append(get_size(gt["bbox"]))

    sizes = np.concatenate(sizes)
    dets = np.concatenate(dets)
    return pd.DataFrame({"Accuracy": dets, "Insect_size": sizes})


def compute_all_size_df(df):
    """
        compute all insect size & accuracy dataframe
        - df: pd.DataFrame({"Accuracy", "Insect_size"})
    """
    df["order"] = df["Insect_size"].apply(lambda x: np.floor(np.log2(x)))
    df2 = df.groupby("order").apply(np.mean)
    return df2


def compute_class_df(each_label_dic, gt_dict):
    """
        compute accuracy per class and create dataframe
        - each_label_dic: {'insect name': insect label, ...}
        - gt_dict: {file id: {'bbox', 'difficult', 'det', 'default_name'}}
    """
    label_name = [name for name in each_label_dic.keys()]
    all_det = []
    all_default_label = []
    for k,v in gt_dict.items():
        all_det.extend(v["det"])
        all_default_label.extend([each_label_dic[name] for name in v["default_name"]])

    all_det = np.asarray(all_det)
    all_default_label = np.asarray(all_default_label)
    default_label, default_count = np.unique(all_default_label, return_counts=True)
    all_det = all_det.astype("bool")

    correct_label = all_default_label[all_det]
    labels, count = np.unique(correct_label, return_counts=True)
    precision_per_class = np.zeros(len(each_label_dic))
    for i in range(len(labels)):
        precision_per_class[labels[i]] = count[i] / default_count[i]
    label_name = [label_name[i] for i in labels]
    precision_per_class = [precision_per_class[i] for i in labels]
    return pd.DataFrame({"Name": label_name, "Precision_per_class": precision_per_class})


def compute_error_df(each_label_dic, gt_dict):
    """
        compute error count per class and create dataframe
        - each_label_dic: {'insect name': insect label, ...}
        - gt_dict: {file id: {'bbox', 'difficult', 'det', 'default_name'}}
    """
    label_name = [name for name in each_label_dic.keys()]
    all_det = []
    all_default_label = []
    for k,v in gt_dict.items():
        all_det.extend(v["det"])
        all_default_label.extend([each_label_dic[name] for name in v["default_name"]])

    all_det = np.asarray(all_det)
    all_default_label = np.asarray(all_default_label)
    all_det = ~all_det.astype("bool")

    error_label = all_default_label[all_det]
    labels, count = np.unique(error_label, return_counts=True)
    error_count = np.zeros(len(each_label_dic))
    for i in range(len(labels)):
        error_count[labels[i]] = count[i]
    label_name = [label_name[i] for i in labels]
    error_count = [error_count[i] for i in labels]
    return pd.DataFrame({"Name": label_name, "Error_count": error_count})