import numpy as np
import torch


def accuracy(model, x, y, bs, return_correction_term=False):
    """
        accuracy for all class
        - model <torch.model>
        - x <Tensor[float, float, float, float]> : Tensor[image_num, channels, width, height]
        - y <Tensor[int]> : Tensor[label]
        - bs <int>
        - return_correction_term <bool> : (default=False)
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
    if return_correction_term is False:
        return correct.mean()
    else:
        return correct.mean(), get_correction_term(result_a, y)


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


def get_correction_term(result_a, y):
    """
        get correction term for loss function
        - result_a <Array[int]> : Array[label]
        - y <Array[int]> : Array[label]
    """
    correction_term = []
    idx, count = np.unique(y, return_counts=True)
    for class_id in idx:
        msk = y == class_id
        result_a_per_class = result_a[msk]
        y_per_class = y[msk]
        correction_term.append(1 / (result_a_per_class == y_per_class).mean())
    return torch.from_numpy(np.asarray(correction_term)).cuda().float()