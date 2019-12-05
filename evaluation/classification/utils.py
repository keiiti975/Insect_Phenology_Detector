import numpy as np


def get_index_from_class_label(class_labels, target_label):
    indexs = []
    for i, class_label in enumerate(class_labels):
        if class_label == target_label:
            indexs.append(i)
    return indexs

def get_size_list_from_xte(xte):
    size_list = []
    for img in xte:
        img = img.transpose(1,2,0)
        size = get_size_from_cropped_img(img)
        size_list.append(size)
    return size_list

def get_size_from_cropped_img(img):
    mask_x, mask_y = np.where(img[:, :, 0] > 0)
    crop_img = img[mask_x[0]:mask_x[-1], mask_y[0]:mask_y[-1], :]
    return crop_img.shape[0] * crop_img.shape[1]

def get_masked_index_from_mask(mask):
    masked_index = []
    for index, label_result in enumerate(mask):
        if label_result == True:
            masked_index.append(index)
    return np.asarray(masked_index)