import numpy as np
import torch
from tqdm import tqdm


def test_classification(model, test_dataloader, mean_feature=None, covariance_metrix=None):
    """
        classify test data
        Args:
            - model: pytorch model
            - test_dataloader: torchvision dataloader
            - mean_feature: list(torch.Tensor(dtype=float)), shape==[class_num]
                , use if classify with mahalonobis distance
            - covariance_metrix: torch.Tensor(dtype=float), shape==[class_num, param_num, param_num]
                , use if classify with mahalonobis distance
    """
    model.eval()
    result_lbl = []
    if mean_feature is not None and covariance_metrix is not None:
        for x in tqdm(test_dataloader, leave=False):
            x = x.cuda()
            feature = model(x).cpu().detach()
            if len(feature.shape) == 1:
                feature = feature[None, :]

            result = torch.zeros(len(feature), len(mean_feature))
            for batch_id in range(len(feature)):
                for class_id in range(len(mean_feature)):
                    feature_difference = feature[batch_id] - mean_feature[class_id]
                    mahalanobis_dist = torch.mm(covariance_metrix[class_id], feature_difference[:, None])
                    mahalanobis_dist = torch.mm(feature_difference[None, :], mahalanobis_dist)
                    result[batch_id][class_id] = -1 / 2 * mahalanobis_dist

            result = torch.nn.functional.softmax(result, dim=1)
            result = torch.max(result, 1)[1]
            result = result.numpy()
            result_lbl.extend(result)
    else:
        for x in test_dataloader:
            x = x.cuda()
            out = model(x)
            if len(out.shape) == 1:
                out = out[None, :]
            result = torch.max(out, 1)[1]
            result = result.cpu().numpy()
            result_lbl.extend(result)
            
    return np.asarray(result_lbl)


def get_mean_feature(model, valid_dataloader, maha_training=False):
    """
        calculate mean feature from validation dataset
        this is used when classify with mahalonobis distance
        Args:
            - model: pytorch model
            - valid_dataloader: torchvision dataloader
    """
    x_id = 0
    feature_per_class = [[] for i in range(model.n_class)]
    mean_feature = [[] for i in range(model.n_class)]
    if maha_training is True:
        labels = valid_dataloader.dataset.sampled_labels
    else:
        labels = valid_dataloader.dataset.labels
        
    for x in tqdm(valid_dataloader, leave=False): 
        x = x.cuda()
        feature = model(x).cpu().detach()
        if len(feature.shape) == 1:
            feature = feature[None, :]
        
        for elem_feature in feature:
            label = labels[x_id]
            feature_per_class[label].append(elem_feature)
            x_id += 1
    
    for i in range(model.n_class):
        mean_feature[i] = torch.stack(feature_per_class[i]).mean(dim=0)
    
    return mean_feature


def estimate_covariance_metrix(mean_feature, model, valid_dataloader, beta=0.1, maha_training=False):
    """
        estimate covariance matrix from mean_feature and model output
        Args:
            - mean_feature: list(torch.Tensor(dtype=float)), shape == [num_classes]
            - model: pytorch model
            - valid_dataloader: torchvision dataloader
            - beta: float, regularization parameter
    """
    x_id = 0
    covariance_metrix = torch.zeros(model.n_class, len(mean_feature[0]), len(mean_feature[0]))
    if maha_training is True:
        labels = valid_dataloader.dataset.sampled_labels
    else:
        labels = valid_dataloader.dataset.labels
    
    idx, count = np.unique(labels, return_counts=True)
    for x in tqdm(valid_dataloader, leave=False):
        x = x.cuda()
        feature = model(x).cpu().detach()
        if len(feature.shape) == 1:
            feature = feature[None, :]
        
        for elem_feature in feature:
            label = labels[x_id]
            feature_difference = elem_feature - mean_feature[label]
            covariance_metrix[label] += torch.mm(feature_difference[:, None], feature_difference[None, :])
            x_id += 1
    
    for i in range(model.n_class):
        covariance_metrix[i] /= count[i]
        
    covariance_metrix[i] += beta * torch.eye(len(mean_feature[0]))
    covariance_metrix[i] = torch.inverse(covariance_metrix[i])
    
    return covariance_metrix