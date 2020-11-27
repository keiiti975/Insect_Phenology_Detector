import cv2
import numpy as np


def compute_padding(coord, delta=100, use_segmentation_lbl=False):
    """
        compute padding size
        - coord: [x1, y1, x2, y2]
        - delta: int
    """
    xmin, ymin, xmax, ymax = coord
    padleft = (2*delta - (xmax - xmin)) // 2
    padright = 2*delta - padleft - (xmax - xmin)
    padtop = (2*delta - (ymax - ymin)) // 2
    padbottom = 2*delta - padtop - (ymax - ymin)
    if use_segmentation_lbl is True:
        return ((padtop,padbottom), (padleft,padright))
    else:
        return ((padtop,padbottom), (padleft,padright), (0,0))

def check_coord(coord, size=200):
    """
        check coordination and correct (width, height)
        - coord: [x1, y1, x2, y2]
        - size: int
    """
    xmin, ymin, xmax, ymax = coord
    if (xmax - xmin) > size:
        xc = (xmin+xmax)//2
        xmin, xmax = xc-(size//2), xc+(size//2)
    if (ymax - ymin) > size:
        yc = (ymin+ymax)//2
        ymin, ymax = yc-(size//2), yc+(size//2)
    return xmin, ymin, xmax, ymax

def crop_standard(img, coord, delta=100):
    """
        standard crop and padding constant
        - img: image_data, np.array
        - coord: [x1, y1, x2, y2]
        - delta: int
    """
    xmin, ymin, xmax, ymax = coord
    xc, yc = (xmin+xmax)//2, (ymin+ymax)//2
    img = img[max(0,yc-delta):yc+delta, max(0,xc-delta):xc+delta]
    xs, ys, _ = img.shape
    padleft = 2*delta - xs
    padtop  = 2*delta - ys
    if padleft or padtop:
        padright  = 2*delta - xs - padleft // 2
        padbottom = 2*delta - ys - padtop // 2
        pad = ((padleft//2,padright),(padtop//2,padbottom), (0,0))
        img = np.pad(img, pad, "constant")
    return img[None,:]

def crop_adjusted(img, coord, delta=100, use_integer_coord=False):
    """
        adjusting crop and padding constant
        - img: image_data, np.array
        - coord: [x1, y1, x2, y2]
        - delta: int
    """
    coord = check_coord(coord)
    xmin, ymin, xmax, ymax = coord
    if use_integer_coord is True:
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
    img = img[ymin:ymax, xmin:xmax].copy()
    padding = compute_padding((0, 0, img.shape[1], img.shape[0]))
    img = np.pad(img, padding, "constant")
    return img[None,:]

def crop_adjusted_std(img, coord, delta=100, use_integer_coord=False):
    """
        adjusting crop and padding constant and std
        - img: image_data, np.array
        - coord: [x1, y1, x2, y2]
        - delta: int
    """
    coord = check_coord(coord)
    xmin, ymin, xmax, ymax = coord
    if use_integer_coord is True:
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
    img = (img - np.mean(img, keepdims=True))/np.std(img, keepdims=True)*32+128
    img = img[ymin:ymax, xmin:xmax].copy()
    padding = compute_padding((0, 0, img.shape[1], img.shape[0]))
    img = np.pad(img, padding, "constant")
    return img[None,:]


def crop_adjusted_std_resize(img, coord, delta=100, use_integer_coord=False):
    """
        adjusting crop and padding constant and std and resize to (200*200)
        - img: image_data, np.array
        - coord: [x1, y1, x2, y2]
        - delta: int
        - use_integer_coord: bool
    """
    coord = check_coord(coord)
    xmin, ymin, xmax, ymax = coord
    if use_integer_coord is True:
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
    if (xmax - xmin) > (ymax - ymin):
        max_length_axis = xmax - xmin
    else:
        max_length_axis = ymax - ymin
    img = (img - np.mean(img, keepdims=True))/np.std(img, keepdims=True)*32+128
    img = img[ymin:ymax, xmin:xmax].copy()
    img = cv2.resize(img, dsize=(200, 200), interpolation=cv2.INTER_LINEAR)
    return img[None,:]


def crop_adjusted_std_resizeFAR(img, coord, delta=100, use_integer_coord=False):
    """
        adjusting crop and padding constant and std and resizeFAR
        FAR: Fix Aspect Ratio
        - img: image_data, np.array
        - coord: [x1, y1, x2, y2]
        - delta: int
        - use_integer_coord: bool
    """
    coord = check_coord(coord)
    xmin, ymin, xmax, ymax = coord
    if use_integer_coord is True:
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
    if (xmax - xmin) > (ymax - ymin):
        max_length_axis = xmax - xmin
    else:
        max_length_axis = ymax - ymin
    img = (img - np.mean(img, keepdims=True))/np.std(img, keepdims=True)*32+128
    img = img[ymin:ymax, xmin:xmax].copy()
    if (xmax - xmin) > (ymax - ymin):
        img = cv2.resize(img, dsize=(200, (int)(img.shape[0]*200/max_length_axis)), interpolation=cv2.INTER_LINEAR)
    else:
        img = cv2.resize(img, dsize=((int)(img.shape[1]*200/max_length_axis), 200), interpolation=cv2.INTER_LINEAR)
    padding = compute_padding((0, 0, img.shape[1], img.shape[0]))
    img = np.pad(img, padding, "constant")
    return img[None,:]