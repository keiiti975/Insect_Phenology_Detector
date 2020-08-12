import numpy as np
import pandas as pd


def compute_anno_stats(anno):
    """
        compute annotation stats
        - anno: {file id: [(insect name, [x1, y1, x2, y2]), ...]}
    """
    C,H,W = [], [], []
    center = []
    idx = []

    for k,v in anno.items():
        for item in v:
            c = item[0]
            xmin, ymin, xmax, ymax = item[1]
            w = xmax - xmin
            h = ymax - ymin
            
            H.append(h)
            W.append(w)
            C.append(c)
            center.append(((xmin+xmax)//2, (ymin+ymax)//2))
            idx.append(k)
            
    H = np.array(H)
    W = np.array(W)
    C = np.array(C)
    center = np.array(center)
    idx = np.array(idx)
    return H,W,C,center,idx


def compute_average_size(H,W,C):
    """
        compute H*W average per class
        - H: [int, ...]
        - W: [int, ...]
        - C: [str, ...]
    """
    idx, count = np.unique(C, return_counts=True)
    map_idx = {k:i for i,k in enumerate(idx)}
    sum_S = np.zeros(idx.shape[0])
    for i in range(len(H)):
        sum_S[map_idx[C[i]]] += H[i] * W[i]
    sum_S_mean = sum_S / count
    return sum_S_mean, idx


def compute_size_correction(H,W,C):
    """
        compute width (or height) size correction
        - H: [int, ...]
        - W: [int, ...]
        - C: [str, ...]
    """
    avg_size, labels = compute_average_size(H,W,C)
    size_correction = np.ones(avg_size.shape[0])
    avg_size = avg_size[:-4]
    avg_size = avg_size/avg_size.max()
    avg_size = 1/avg_size
    size_correction[:-4] = avg_size ** 0.5
    size_correction = {label:resize_size for resize_size, label in zip(size_correction, labels)}
    return size_correction


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