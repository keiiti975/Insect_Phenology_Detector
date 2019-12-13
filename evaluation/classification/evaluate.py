import numpy as np
import torch


def accuracy(model, x, y, bs):
    """
    accuracy for all class
    """
    model.eval()
    result_a = []
    y_sum = len(y)
    for i in range(0, y_sum - bs, bs):
        x2 = x[i:i + bs]
        out = model(x2)
        result = torch.max(out, 1)[1]
        result = result.cpu().numpy()
        result_a.extend(result)

    i = i + bs
    x2 = x[i:]
    out = model(x2)
    result = torch.max(out, 1)[1]
    result = result.cpu().numpy()
    result_a.extend(result)

    result_a = np.asarray(result_a)
    y = y.cpu().numpy()
    correct = result_a == y
    model.train()
    return correct.mean()


def confusion_matrix(model, x, y, labels, bs):
    model.eval()
    result_c = []
    y_sum = len(y)
    for i in range(0, y_sum - bs, bs):
        x2 = x[i:i + bs]
        out = model(x2)
        result = torch.max(out, 1)[1]
        result = result.cpu().numpy()
        result_c.extend(result)

    i = i + bs
    x2 = x[i:i + bs]
    out = model(x2)
    result = torch.max(out, 1)[1]
    result = result.cpu().numpy()
    result_c.extend(result)

    result_c = np.asarray(result_c)
    y = y.cpu().numpy()
    confusion_matrix = []
    for i in range(len(labels)):
        msk = np.isin(y, i)
        class_estimation = []
        for j in range(len(labels)):
            estimation = result_c[msk] == j
            class_estimation.append(estimation.sum())
        confusion_matrix.append(class_estimation)
    return np.asarray(confusion_matrix).astype("float")