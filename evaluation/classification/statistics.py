import numpy as np
import pandas as pd


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
    df2 = pd.DataFrame({"order": df2["order"].values, "Accuracy": df2["Accuracy"].values, "Insect_size": df2["Insect_size"].values})
    return df2


def get_size_list_from_xte(xte):
    size_list = []
    for img in xte:
        img = img.transpose(1,2,0)
        size = get_size_from_cropped_img(img)
        size_list.append(size)
    return size_list


def get_size_from_cropped_img(img):
    mask_x, mask_y = np.where(img[:, :, 0] > 0)
    crop_img = img[mask_x[0]:mask_x[-1], mask_y[0]:mask_y[-1], :]
    return crop_img.shape[0] * crop_img.shape[1]