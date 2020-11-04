import numpy as np
import torch
from model.resnet.predict import test_classification


def accuracy(model, test_dataloader, return_correct=False, mean_feature=None, covariance_metrix=None):
    """
        calculate accuracy for all class
        Args:
            - model: pytorch model
            - test_dataloader: torchvision dataloader
            - return_correct: bool, used for visualizing failed data
            - mean_feature: list(torch.Tensor(dtype=float)), shape==[class_num]
                , use if classify with mahalonobis distance
            - covariance_metrix: torch.Tensor(dtype=float), shape==[class_num, param_num, param_num]
                , use if classify with mahalonobis distance
    """
    result_a = test_classification(model, test_dataloader, mean_feature, covariance_metrix)
    y = test_dataloader.dataset.labels
    correct = result_a == y
    if return_correct is True:
        return correct.mean(), correct
    else:
        return correct.mean()
    

def confusion_matrix(model, test_dataloader, labels, mean_feature=None, covariance_metrix=None):
    """
        calculate confusion metrix
        Args:
            - model: pytorch model
            - test_dataloader: torchvision dataloader
            - labels: [str, ...], insect label
            - mean_feature: list(torch.Tensor(dtype=float)), shape==[class_num]
                , use if classify with mahalonobis distance
            - covariance_metrix: torch.Tensor(dtype=float), shape==[class_num, param_num, param_num]
                , use if classify with mahalonobis distance
    """
    result_c = test_classification(model, test_dataloader, mean_feature, covariance_metrix)
    y = test_dataloader.dataset.labels
    confusion_matrix = []
    for i in range(len(labels)):
        msk = np.isin(y, i)
        class_estimation = []
        for j in range(len(labels)):
            estimation = result_c[msk] == j
            class_estimation.append(estimation.sum())
        confusion_matrix.append(class_estimation)
    return np.asarray(confusion_matrix).astype("float")