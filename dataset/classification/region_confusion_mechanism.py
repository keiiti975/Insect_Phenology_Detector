import numpy as np
from numpy.random import *


def compute_padding(xmax, ymax, delta=100):
    """
        return padding size
    """
    padleft = (2 * delta - xmax) // 2
    padright = 2 * delta - padleft - xmax
    padtop = (2 * delta - ymax) // 2
    padbottom = 2 * delta - padtop - ymax
    return ((padtop, padbottom), (padleft, padright), (0, 0))


def crop_adjusted(img, delta=100):
    """
        adjusting crop and padding constant
    """
    ymax, xmax, _ = img.shape
    img = img[0:ymax, 0:xmax].copy()
    padding = compute_padding(xmax, ymax)
    img = np.pad(img, padding, "constant")
    return img


def region_confusion_mechanism(xtr, division_number=7, neighborhood_range=1):
    new_imgs = []
    new_coordinates = []
    for k in range(xtr.shape[0]):
        output = region_confusion_mechanism_(
            xtr[k], division_number=division_number, neighborhood_range=neighborhood_range)
        if (output[0].shape == (200, 200, 3)):
            new_imgs.append(output[0])
            new_coordinates.append(output[1])
    return np.asarray(new_imgs), np.asarray(new_coordinates)


def region_confusion_mechanism_(img, division_number=7, neighborhood_range=1):
    sample_img = img
    mask_x, mask_y = np.where(sample_img[:, :, 0] > 0)
    if len(mask_x) == 0 and len(mask_y) == 0:
        # sometime image value == 0
        return img
    crop_img = sample_img[mask_x[0]:mask_x[-1], mask_y[0]:mask_y[-1], :]

    default_img_y, default_img_x, _ = crop_img.shape
    subregion_width = (int)(default_img_x / division_number)
    subregion_height = (int)(default_img_y / division_number)
    div_img = []
    for i in range(division_number):
        for j in range(division_number):
            div_img.append(crop_img[i * subregion_height:(i + 1) * subregion_height,
                                    j * subregion_width:(j + 1) * subregion_width, :])

    permutation_of_region_width = []
    permutation_of_region_height = []
    for i in range(division_number):
        rand_num = randint(-1 * neighborhood_range, neighborhood_range,
                           division_number) + np.arange(division_number)
        permutation_of_region_width.append(rand_num.argsort())
    for i in range(division_number):
        rand_num = randint(-1 * neighborhood_range, neighborhood_range,
                           division_number) + np.arange(division_number)
        permutation_of_region_height.append(rand_num.argsort())
    permutation_of_region_width = np.asarray(permutation_of_region_width)
    permutation_of_region_height = np.asarray(permutation_of_region_height)

    new_coordinate = []
    for i in range(division_number):
        for j in range(division_number):
            new_coordinate.append(
                permutation_of_region_height[j, i] * division_number + permutation_of_region_width[i, j])
    new_coordinate = np.asarray(new_coordinate)

    for i in range(division_number):
        if i == 0:
            new_crop_img = np.concatenate(
                [div_img[new_coordinate[i * division_number + j]] for j in range(division_number)])
        else:
            new_crop_img = np.concatenate([new_crop_img, np.concatenate(
                [div_img[new_coordinate[i * division_number + j]] for j in range(division_number)])], axis=1)

    new_img = crop_adjusted(new_crop_img)
    return (new_img, new_coordinate)
