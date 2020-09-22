import torch


def define_weight(counts):
    """
        get class weight from class count
        - counts: [int, ...]
    """
    counts = counts / counts.sum()
    weights = torch.from_numpy(1 / counts).cuda().float()
    return weights


def min_euclidean(out, sem):
    """
        unused
        pytorch calculate euclidean
    """
    nbr = sem.size(1)
    ab = torch.mm(out.view(-1, nbr), sem.t())
    ab = ab.view(out.size(0), out.size(1), sem.size(0))
    aa = (sem**2).sum(1)
    bb = (out**2).sum(-1)
    res = aa[None, None, :] + bb[:, :, None] - 2 * ab
    return res.min(-1)[1]


def predict_label_from_semantic(predict_semantic, semantic_vectors):
    """
        unused
    """
    predict_labels = []
    for p_semantic in predict_semantic:
        p_semantic = p_semantic[:, None, None].transpose(0, 1).transpose(1, 2)
        label = min_euclidean(p_semantic, semantic_vectors)
        predict_labels.append(int(label.cpu().numpy()))
    return predict_labels
