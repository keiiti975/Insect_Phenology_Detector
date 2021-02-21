# -*- coding: utf-8 -*-
import numpy as np
import torch
from model.resnet.predict import test_classification


def accuracy(test_dataloader, model, return_correct=False):
    """
        正解率の計算
        引数:
            - test_dataloader: データローダ
            - model: モデル
            - return_correct: bool, 実験結果(correct)を返す
    """
    result_a = test_classification(test_dataloader, model)
    y = test_dataloader.dataset.labels
    correct = result_a == y
    if return_correct is True:
        return correct.mean(), correct
    else:
        return correct.mean()
    

def confusion_matrix(test_dataloader, model, labels):
    """
        混同行列の計算
        引数:
            - test_dataloader: データローダ
            - model: モデル
            - labels: [str, ...], 昆虫ラベル, 順番注意!
    """
    result_c = test_classification(model, test_dataloader)
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