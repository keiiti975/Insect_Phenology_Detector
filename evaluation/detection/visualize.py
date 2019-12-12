import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join as pj
import pandas as pd

    
def plot_df_distrib_size(df, figure_root, save=False):
    """
        plot detection accuracy distribution of size
        - df: pd.DataFrame({"Accuracy", "Insect_size"})
        - figure_root: str
        - save: bool
    """
    df.plot(x="Insect_size", y="Accuracy", logx=True, legend=False)
    plt.title("Accuracy_distribution_of_size")
    plt.ylabel("Accuracy")
    plt.ylim(0.7, 1.01)
    if save is True:
        os.makedirs(figure_root, exist_ok=True)
        plt.savefig(pj(figure_root, "accuracy_distribution_of_size.png"), bbox_inches="tight")

        
def plot_df_distrib_class(df, figure_root, save=False, color="green"):
    """
        plot detection accuracy distribution of class
        - df: pd.DataFrame({"Name", "Precision_per_class"})
        - figure_root: str
        - save: bool
        - color: str
    """
    df.plot(kind="bar", x="Name", y="Precision_per_class", legend=False, color="green")
    plt.title("Accuracy_distribution_of_class")
    plt.ylabel("Accuracy")
    plt.ylim(0., 1.)
    if save is True:
        os.makedirs(figure_root, exist_ok=True)
        plt.savefig(pj(figure_root, "accuracy_distribution_of_class.png"), bbox_inches="tight")
        

def plot_df_error(df, figure_root, save=False, color="green"):
    """
        plot detection error count per class
        - df: pd.DataFrame({"Name", "Error_count"})
        - figure_root: str
        - save: bool
        - color: str
    """
    df.plot(kind="bar", x="Name", y="Error_count", legend=False, color="green")
    plt.title("Error_count_per_class")
    plt.ylabel("Error_count")
    if save is True:
        os.makedirs(figure_root, exist_ok=True)
        plt.savefig(pj(figure_root, "error_count_per_class.png"), bbox_inches="tight")
    

def plot_pr_curve(df, figure_root, save=False):
    """
        plot precision-recall curve
        - df: pd.DataFrame({"precision", "recall"})
        - figure_root: str
        - save: bool
    """
    df.plot(x="recall", y="precision", legend=False)
    plt.title("PR_Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    if save is True:
        os.makedirs(figure_root, exist_ok=True)
        plt.savefig(pj(figure_root, "pr_curve.png"), bbox_inches="tight")
    

def vis_detections(im, dets, class_name="insects", color_name="green", thresh=0.3):
    """
        Visual debugging of detections
    """
    color = {"blue": (0, 0, 204), "green": (0, 204, 0), "red": (204, 0, 0)}[color_name]
    for i in range(dets.shape[0]):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], color, 2)
            cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
    return im


def xx():
    """
        unused
    """
    s = pd.Series(dets, index=sizes)
    plt.hist(np.log10(s[s == 1].index.tolist()), bins=100, alpha=.5)
    plt.hist(np.log10(s[s == 0].index.tolist()), bins=100, alpha=.5)