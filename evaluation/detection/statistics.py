import numpy as np
import pandas as pd


def compute_each_size_df(eval_metrics):
    """
        compute each insect size & accuracy dataframe
        - result: Voc_Evaluater output
    """
    all_label_gt_size = []
    all_label_gt_result = []
    for eval_metric in eval_metrics:
        all_label_gt_size.extend(eval_metric["gt_size"])
        all_label_gt_result.extend(eval_metric["gt_result"])
    return pd.DataFrame({"Accuracy": np.asarray(all_label_gt_result), "Insect_size": np.asarray(all_label_gt_size)})


def compute_all_size_df(each_df):
    """
        compute all insect size & accuracy dataframe
        - each_df: pd.DataFrame({"Accuracy", "Insect_size"})
    """
    each_df["order"] = each_df["Insect_size"].apply(lambda x: np.floor(np.log2(x)))
    df = each_df.groupby("order").apply(np.mean)
    df = pd.DataFrame({"order": df["order"].values, "Accuracy": df["Accuracy"].values, "Insect_size": df["Insect_size"].values})
    return df