# -*- coding: utf-8 -*-
import numpy as np
import torch
from tqdm import tqdm


def test_classification(test_dataloader, model):
    """
        分類する
        引数:
            - test_dataloader: データローダ
            - model: モデル
    """
    model.eval()
    result_lbl = []
    for x in test_dataloader:
        x = x.cuda()
        out = model(x)
        if len(out.shape) == 1:
            # bsが1だとエラーとなるので, 空の次元を挿入
            out = out[None, :]
        result = torch.max(out, 1)[1]
        result = result.cpu().numpy()
        result_lbl.extend(result)
            
    return np.asarray(result_lbl)