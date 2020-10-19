import numpy as np
import torch


def accuracy(model, test_dataloader, return_correct=False):
    """
        calculate accuracy for all class
        Args:
            - model: pytorch model
            - test_dataloader: torchvision dataloader
            - return_correct: bool, used for visualizing failed data
    """
    model.eval()
    result_a = []
    for x in test_dataloader:
        x = x.cuda()
        out = model(x)
        if len(out.shape) == 1:
            out = out[None, :]
        result = torch.max(out, 1)[1]
        result = result.cpu().numpy()
        result_a.extend(result)

    result_a = np.asarray(result_a)
    y = test_dataloader.dataset.labels
    correct = result_a == y
    model.train()
    if return_correct is True:
        return correct.mean(), correct
    else:
        return correct.mean()
    

def confusion_matrix(model, test_dataloader, labels):
    """
        calculate confusion metrix
        Args:
            - model: pytorch model
            - test_dataloader: torchvision dataloader
            - labels: [str, ...], insect label
    """
    model.eval()
    result_c = []
    for x in test_dataloader:
        x = x.cuda()
        out = model(x)
        if len(out.shape) == 1:
            out = out[None, :]
        result = torch.max(out, 1)[1]
        result = result.cpu().numpy()
        result_c.extend(result)

    result_c = np.asarray(result_c)
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