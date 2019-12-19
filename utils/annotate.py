import numpy as np


def annotate_center(target, coord):
    """
        unused
    """
    xmin, ymin, xmax, ymax = coord
    xc, yc = (xmin+xmax)//2, (ymin+ymax)//2
    target[yc,xc]=1
    
    
def annotate_full(target, coord):
    """
        unused
    """
    xmin, ymin, xmax, ymax = coord
    target[ymin:ymax,xmin:xmax]=1
    
    
def annotate_gaussian(target, coord):
    """
        unused
    """
    fill_values = None #create_gaussian
    target[ymin:ymax,xmin:xmax]=fill_values
    

def make_detect_target(img, coords, anno_func=annotate_full):
    """
        unused
    """
    target = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    for coord in coords:
        anno_func(target, coord)
    return target