from os.path import join as pj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from evaluation.classification.utils import get_size_list_from_xte


def compute_each_size_df(result, xte, yte):
    """
        compute each insect size & accuracy dataframe
        - result: model label output
        - xte: test images
        - yte: test labels
    """
    xte_size = np.asarray(get_size_list_from_xte(xte))
    mask = result == yte
    return pd.DataFrame({"Accuracy": mask, "Insect_size": xte_size})

def compute_all_size_df(df):
    """
        compute all insect size & accuracy dataframe
        - df: pd.DataFrame({"Accuracy", "Insect_size"})
    """
    df["order"] = df["Insect_size"].apply(lambda x: np.floor(np.log2(x)))
    df2 = df.groupby("order").apply(np.mean)
    return df2