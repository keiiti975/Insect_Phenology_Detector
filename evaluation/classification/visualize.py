# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pj


def plot_size_of_anno(W,H):
    """
        plot size (scatter, H, W) of anno
        - H: [int, ...]
        - W: [int, ...]
    """
    fig, axes = plt.subplots(1,3, figsize=(30,10))
    axes[0].scatter(W,H, alpha=0.5)
    axes[1].hist(H)
    axes[2].hist(W)
    

def plot_size_by_class_of_anno(H,W,C):
    """
        plot size scatter of anno, by class
        - H: [int, ...]
        - W: [int, ...]
        - C: [str, ...]
    """
    fig, axes = plt.subplots(1,1, figsize=(10,10))
    for c in np.unique(C):
        msk = C==c
        h = H[msk]
        w = W[msk]
        axes.scatter(w,h, alpha=.5, label=c)
    fig.legend()
    

def create_confusion_matrix(matrix, ntests, labels, figure_root, save=False):
    """
        混同行列の描画
        引数:
            - matrix: np.array, 混同行列
            - ntests: np.array, 各クラスの個体数
            - labels: [str, ...], 昆虫ラベル
            - figure_root: str, 図の保存先フォルダのパス
            - save: bool, 図を保存するかどうか
    """
    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.grid"] = False
    fig, axe = plt.subplots(1, 1, figsize=(10, 10))
    for i in range(len(labels)):
        for j in range(len(labels)):
            matrix[i, j] = matrix[i, j] / ntests[i]
    axe.set_xticks(np.arange(len(labels)))
    axe.set_yticks(np.arange(len(labels)))
    axe.set_xticklabels(labels)
    axe.set_yticklabels(labels)
    axe.imshow(matrix)
    plt.setp(axe.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")
    matrix2 = np.round(matrix, 2)
    for i in range(len(labels)):
        for j in range(len(labels)):
            axe.text(j, i, matrix2[i, j], size=20,
                     ha="center", va="center", color="black")
    axe.set_title("Confusion_matrix")
    axe.set_xlabel("Output_class")
    axe.set_ylabel("Target_class")
    if save is True:
        plt.savefig(pj(figure_root, "confusion_matrix.png"), bbox_inches="tight")

        
def plot_df_distrib_size(df, figure_root, save=False):
    """
        サイズ帯(=log2 Insect_size)ごとのRecallを描画
        この関数での描画は不十分なので, resultフォルダにある関数を使ってほしい
        引数:
            - df: pd.DataFrame({"order", "Accuracy", "Insect_size"})
            - figure_root: str, 図の保存先フォルダのパス
            - save: bool, 図を保存するかどうか
    """
    df.plot(x="Insect_size", y="Recall", logx=True, legend=False)
    plt.title("Recall with size distribution")
    plt.ylabel("Recall")
    plt.ylim(0.7, 1.01)
    if save is True:
        plt.savefig(pj(output_dir, "recall_with_size_distribution.png"), bbox_inches="tight")