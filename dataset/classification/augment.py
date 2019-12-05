import numpy as np
from scipy import ndimage
from tqdm import tqdm


def data_augment(X, Y, rotate):
    imgs, lbls = [], []
    for img, lbl in tqdm(zip(X, Y), total=Y.shape[0], leave=False):
        for i in range(0, 360, rotate):
            x = ndimage.rotate(img, i, reshape=False)
            lbls.append(lbl)
            imgs.append(x)
    return np.array(imgs), np.array(lbls)


def data_augment_DCL(X, Y, target_dest_or_not, target_coordinate, rotate):
    imgs, lbls, dest_lbls, coord_lbls = [], [], [], []
    for img, lbl, dest_lbl, coord_lbl in tqdm(zip(X, Y, target_dest_or_not, target_coordinate), total=Y.shape[0], leave=False):
        for i in range(0, 360, rotate):
            x = ndimage.rotate(img, i, reshape=False)
            lbls.append(lbl)
            imgs.append(x)
            dest_lbls.append(dest_lbl)
            coord_lbls.append(coord_lbl)
    return np.array(imgs), np.array(lbls), np.array(dest_lbls), np.array(coord_lbls)
