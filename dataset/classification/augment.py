import numpy as np
from PIL import Image
from scipy import ndimage
from torchvision.transforms.functional import resize, crop
from tqdm import tqdm


def adopt_rotate(X, Y, rotate):
    imgs, lbls = [], []
    for img, lbl in tqdm(zip(X, Y), total=Y.shape[0], leave=False):
        for i in range(0, 360, rotate):
            x = ndimage.rotate(img, i, reshape=False)
            lbls.append(lbl)
            imgs.append(x)
    return np.array(imgs), np.array(lbls)


def adopt_rotate_DCL(X, Y, target_dest_or_not, target_coordinate, rotate):
    imgs, lbls, dest_lbls, coord_lbls = [], [], [], []
    for img, lbl, dest_lbl, coord_lbl in tqdm(zip(X, Y, target_dest_or_not, target_coordinate), total=Y.shape[0], leave=False):
        for i in range(0, 360, rotate):
            x = ndimage.rotate(img, i, reshape=False)
            lbls.append(lbl)
            imgs.append(x)
            dest_lbls.append(dest_lbl)
            coord_lbls.append(coord_lbl)
    return np.array(imgs), np.array(lbls), np.array(dest_lbls), np.array(coord_lbls)


def adopt_random_size_crop(xtr, low=0, high=50, interpolation=Image.BILINEAR):
    """
        adopt random size crop
        - xtr <Array[int, int, int, int]> : Array[image_num, width, height, channels]
        - low <int>
        - hight <int>
        - interpolation <int> : etc. Image.BILINEAR
    """
    new_xtr = []
    for elem_xtr in xtr:
        image = Image.fromarray(np.uint8(elem_xtr))
        trim_randint = np.random.randint(low, high, 1)[0]
        image = crop(image, trim_randint, trim_randint, 200-2*trim_randint, 200-2*trim_randint)
        image = resize(image, (200, 200), interpolation)
        new_xtr.append(np.asarray(image))
    return np.asarray(new_xtr)
