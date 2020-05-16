from math import sqrt as sqrt
from itertools import product as product
import torch


def get_prior_box(image_size, feature_sizes, other_aspect_ratio=[2], clip=True):
    """
        compute prior anchor box.
        used for training and test.
        Args:
            - image_size: int, input image size, choice [320, 512, 1024]
            - feature_sizes: [int, ...], edge size of layer for each layer
            - clip: bool, clamp prior anchor box size to 0~1
    """
    num_priors = len(feature_sizes)
    steps = [image_size / size for size in feature_sizes]
    min_sizes = [4 * step for step in steps]
    mean = []
    for k, feature_size in enumerate(feature_sizes):
        for y, x in product(range(feature_size), repeat=2):
            feature_step_k = image_size / steps[k]
            # unit center x,y
            cx = (x + 0.5) / feature_step_k
            cy = (y + 0.5) / feature_step_k

            # aspect_ratio: 1
            # rel size: min_size
            normalized_min_size_k = min_sizes[k] / image_size
            mean += [cx, cy, normalized_min_size_k, normalized_min_size_k]

            # rest of aspect ratios
            for ar in other_aspect_ratio:
                mean += [cx, cy, normalized_min_size_k*sqrt(ar), normalized_min_size_k/sqrt(ar)]
                mean += [cx, cy, normalized_min_size_k/sqrt(ar), normalized_min_size_k*sqrt(ar)]
    # back to torch land
    output = torch.Tensor(mean).view(-1, 4)
    if clip:
        output.clamp_(max=1, min=0)
    return output
