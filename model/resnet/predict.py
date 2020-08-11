import numpy as np
import torch


def test_classification(model, imgs, bs=20):
    model.eval()
    result_lbl = []
    img_num = len(imgs)
    i = 0
    while i + bs < img_num:
        x = imgs[i:i + bs]
        out = model(x)
        result = torch.max(out, 1)[1]
        result = result.cpu().numpy()
        result_lbl.extend(result)
        i = i + bs

    x = imgs[i:]
    out = model(x)
    if len(out.shape) == 1:
        out = out.unsqueeze(0)
    result = torch.max(out, 1)[1]
    result = result.cpu().numpy()
    result_lbl.extend(result)
    return np.asarray(result_lbl)