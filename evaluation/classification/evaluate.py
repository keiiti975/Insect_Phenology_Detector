import numpy as np
import torch


def accuracy(model, test_dataloader, return_correction_term=False, low_trainable_correction=False):
    """
        calculate accuracy for all class
        Args:
            - model: pytorch model
            - test_dataloader: torchvision dataloader
            - return_correction_term: bool
            - low_trainable_correction: bool
    """
    model.eval()
    result_a = []
    for x, _ in test_dataloader:
        x = x.cuda()
        out = model(x)
        result = torch.max(out, 1)[1]
        result = result.cpu().numpy()
        result_a.extend(result)

    result_a = np.asarray(result_a)
    y = test_dataloader.dataset.labels
    correct = result_a == y
    model.train()
    if return_correction_term is False:
        return correct.mean()
    else:
        return correct.mean(), get_correction_term(result_a, y, low_trainable_correction)


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
    for x, _ in test_dataloader:
        x = x.cuda()
        out = model(x)
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


def get_correction_term(result_a, y, low_trainable_correction=False):
    """
        get correction term for loss function
        Args:
            - result_a: np.array, classification result
            - y: np.array, target labels
            - low_trainable_correction: bool
    """
    correction_term = []
    idx, count = np.unique(y, return_counts=True)
    for class_id in idx:
        msk = y == class_id
        result_a_per_class = result_a[msk]
        y_per_class = y[msk]
        correction_term.append(1 / (result_a_per_class == y_per_class).mean())
    
    if low_trainable_correction == True:
        correction_term = np.asarray(correction_term)
        sorted_correction_term_index = np.argsort(correction_term)
        reverse_index = np.flip(sorted_correction_term_index)
        count = 0
        index_num = len(idx)
        for low_index in reverse_index:
            if count < int(index_num * (2 / 3)):
                correction_term[low_index] = 1.
                count += 1
            else:
                correction_term[low_index] = 0.
                count += 1
            
    return torch.from_numpy(np.asarray(correction_term)).cuda().float()