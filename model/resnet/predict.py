import numpy as np
import torch


def test_classification(model, test_dataloader):
    """
        classify test data
        Args:
            - model: pytorch model
            - test_dataloader: torchvision dataloader
    """
    model.eval()
    result_lbl = []
    for x, _ in test_dataloader:
        x = x.cuda()
        out = model(x)
        result = torch.max(out, 1)[1]
        result = result.cpu().numpy()
        result_lbl.extend(result)
    return np.asarray(result_lbl)