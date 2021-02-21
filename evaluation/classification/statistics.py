# -*- coding: utf-8 -*-
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
        実験結果と昆虫サイズを並べたデータフレームを作成
        引数:
            - result: np.array, 実験結果
            - xte: np.array, テスト用昆虫画像
            - yte: np.array, テスト用ラベル
    """
    xte_size = np.asarray(get_size_list_from_xte(xte)) # 画像から昆虫サイズを逆算
    mask = result == yte
    return pd.DataFrame({"Recall": mask, "Insect_size": xte_size})


def compute_all_size_df(df):
    """
        サイズ帯(=log_2 Insect_size)ごとに実験結果を集計
        引数:
            - df: pd.DataFrame({"Recall", "Insect_size"})
    """
    df["order"] = df["Insect_size"].apply(lambda x: np.floor(np.log2(x)))
    df2 = df.groupby("order").apply(np.mean)
    df2 = pd.DataFrame({"order": df2["order"].values, "Recall": df2["Recall"].values, "Insect_size": df2["Insect_size"].values})
    return df2


def get_size_list_from_xte(xte):
    """
        昆虫画像から昆虫サイズを逆算
        引数:
            - xte: np.array, テスト用昆虫画像
    """
    size_list = []
    for img in xte:
        size = get_size_from_cropped_img(img)
        size_list.append(size)
    return size_list


def get_size_from_cropped_img(img):
    """
        get insect size from test image
        Args:
            - img: np.array, shape == [height, width, channels]
    """
    mask_y, mask_x = np.where(img[:, :, 0] > 0)
    crop_img = img[mask_y[0]:mask_y[-1], mask_x[0]:mask_x[-1], :]
    return crop_img.shape[0] * crop_img.shape[1]


def get_precisions(df):
    """
        compute precision from validation matrix
        Args:
            - df: pd.DataFrame, validation matrix
    """
    precisions = []
    for i in range(len(df)):
        val_count_per_label = df[i:i+1]
        index = val_count_per_label.columns[1:]
        val_count_per_label = [int(val_count_per_label[idx]) for idx in index]
        precision = val_count_per_label[i]/sum(val_count_per_label)
        precisions.append(precision)
    return np.asarray(precisions)


def get_average_precision(df):
    """
        compute average precision from validation matrix
        Args:
            - df: pd.DataFrame, validation matrix
    """
    total_insects = 0
    column_count = len(df.columns) - 1
    for i in range(column_count):
        total_insects += sum(df.iloc[:, i + 1])
        
    true_positive_count = 0
    for i in range(column_count):
        true_positive_count += df.iloc[i, i + 1]
        
    return float(true_positive_count / total_insects)