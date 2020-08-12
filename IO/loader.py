import xml.etree.ElementTree as ET
from os import listdir as ld
from os.path import join as pj
import numpy as np
from PIL import Image
from scipy import sparse
import torch
from tqdm import tqdm_notebook as tqdm

def get_name(child):
    """
    """
    return child.getchildren()[0].text

def get_coords(child):
    """
    """
    bndbox = child.getchildren()[-1]
    return [int(x.text) for x in bndbox.getchildren()]

def get_objects(fp):
    """
    """
    tree = ET.parse(fp)
    x =  filter(lambda x:x.tag=="object", tree.getroot())
    return filter(lambda x:get_name(x) != "body size", x)

def parse_annotations(fp):
    """
        XML annotation file kara
        annotation list shutsuryoku
    """
    return [(get_name(objects), get_coords(objects)) for objects in get_objects(fp)]

def file_id(file_path):
    """
        get file id from file path
        ex. ~/20180614-1959.xml -> 20180614-1959
        - file_path: str
    """
    file_id = file_path.split("/")[-1]
    file_id = file_id.split(".")[0]
    return file_id

def load_path(data_root, img_folder, anno_folders):
    """
        load annotation and image path
        - data_root: str
        - img_folder: str
        - anno_folders: [str, ...]
    """
    annos = []
    for anno_folder in anno_folders:
        annos_name = ld(pj(data_root, anno_folder))
        annos.extend([pj(data_root, anno_folder, x) for x in annos_name])
    imgs  = ld(pj(data_root, img_folder))
    imgs  = [pj(data_root, img_folder, x) for x in imgs]
    return annos, imgs

def load_images(img_paths):
    """
        load images and map to file id
        - img_paths: [str, ...]
    """
    for img_path in img_paths:
        if ".ipynb_checkpoints" in img_path.split("/")[-1]:
            img_paths.remove(img_path)
    images = {file_id(img_path):np.array(Image.open(img_path)) for img_path in img_paths}
    return images

def load_annotations_path(annos, images):
    """
        load annotations path and map to filename
        - annos: [str, ...]
        - images: {file id: image data, np.array}
    """
    annotations_path = {idx: list(filter(lambda x:idx in x, annos)) for idx in images}
    annotations_path = {k:v for  k,v in annotations_path.items() if len(v)>0}
    return annotations_path

def load_images_path(imgs, annotations_path):
    """
        unused
        load images path and map to filename
    """
    images_path = {idx: list(filter(lambda x:idx in x,imgs))[0] for idx in annotations_path}
    return images_path

def load_annotations(annotations_path):
    """
        load annotations
        - annotations_path: {file id: [str]}
    """
    anno = {}
    for k,v in annotations_path.items():
        anno[k]=[]
        for x in filter(lambda x:x.endswith(".xml"), v):
            anno[k].extend(parse_annotations(x))
    return anno

def remove_unused_labels(lbls, counts):
    """
        remove labels such as "Unknown", "]", ...
        Args:
            - lbls: np.array
            - counts: np.array
    """
    unused_labels = ["]"]
    for unused_label in unused_labels:
        unused_filter = lbls != unused_label
        lbls = lbls[unused_filter]
        counts = counts[unused_filter]
    return lbls, counts

def get_label_dic(anno, each_flag=False, make_refinedet_data=False):
    """
        get label_dic from annotations
        - anno: {file id: [(insect name, [x1, y1, x2, y2]), ...]}
        - each_flag: bool
    """
    lbl = []
    for k,v in anno.items():
        for value in v:
            lbl.append(value[0])
    lbl = np.asarray(lbl)
    lbls, counts = np.unique(lbl, return_counts=True)
    lbls, counts = remove_unused_labels(lbls, counts)
    print(lbls)
    print(counts)
    if each_flag is True:
        label_dic = {k:i for i,k in enumerate(lbls)}
    else:
        if make_refinedet_data is False:
            label_dic = {k:1 for k in lbls}
        else:
            label_dic = {k:0 for k in lbls}
    return label_dic

def rank_roidb_ratio(roidb):
    """
        rank roidb based on the ratio between width and height.
    """
    ratio_large = 2 # largest ratio to preserve.
    ratio_small = 0.5 # smallest ratio to preserve.
    
    ratio_list = []
    for i in range(len(roidb)):
        width = roidb[i]['width']
        height = roidb[i]['height']
        ratio = width / float(height)

        if ratio > ratio_large:
            roidb[i]['need_crop'] = 1
            ratio = ratio_large
        elif ratio < ratio_small:
            roidb[i]['need_crop'] = 1
            ratio = ratio_small        
        else:
            roidb[i]['need_crop'] = 0

        ratio_list.append(ratio)

    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    return ratio_list[ratio_index], ratio_index

def make_roidb(anno, label_dic, images_path):
    """
        make roidb
    """
    count = 0
    roidb = []
    for k,v in anno.items():
        boxes = {"boxes": np.array([value[1] for value in anno[k]], dtype="uint16")}
        names = [value[0] for value in anno[k]]
        gt_classes = {"gt_classes": np.array([label_dic[name] for name in names], dtype="int32")}
        overlap = [[1 if i==idx else 0 for i in range(10)] for idx in gt_classes["gt_classes"]]
        gt_overlap = {"gt_overlap": sparse.csr_matrix(np.array(overlap, dtype="float32"))}
        flipped = {"flipped": False}
        img_id = {"img_id": count}
        image = {"image": images_path[k]}
        img = np.array(Image.open(images_path[k]))
        width = {"width": img.shape[1]}
        height = {"height": img.shape[0]}
        need_crop = {"need_crop": 0}
        count += 1

        db = {}
        db.update(boxes)
        db.update(gt_classes)
        db.update(gt_overlap)
        db.update(flipped)
        db.update(img_id)
        db.update(image)
        db.update(width)
        db.update(height)
        db.update(need_crop)
        roidb.append(db)
    return roidb

def make_roi_dataset(anno, label_dic, images_path):
    """
        make roi dataset
    """
    roidb = make_roidb(anno, label_dic, images_path)
    ratio_list, ratio_index = rank_roidb_ratio(roidb)
    return roidb, ratio_list, ratio_index

def get_anno_recs(anno, each_flag=False):
    """
        get annotation recs
    """
    imagenames = []
    recs = {}
    for k,v in anno.items():
        imagenames.append(k)
        rec_image = []
        for i in range(len(v)):
            if each_flag is True:
                name = {"name": v[i][0]}
            else:
                name = {"name": "insects"}
            default_name = {"default_name": v[i][0]}
            difficult = {"difficult": 0}
            bbox = {"bbox": v[i][1]}

            rec = {}
            rec.update(name)
            rec.update(default_name)
            rec.update(difficult)
            rec.update(bbox)
            rec_image.append(rec)
        imagename = {k: rec_image}
        recs.update(imagename)
    return imagenames, recs

def compute_padding(coord, delta=100):
    """
        return padding size
    """
    xmin, ymin, xmax, ymax = coord
    padleft    = (2*delta - (xmax - xmin)) // 2
    padright   = 2*delta - padleft - (xmax - xmin)
    padtop     = (2*delta - (ymax - ymin)) // 2
    padbottom  = 2*delta - padtop - (ymax - ymin)
    return ((padtop,padbottom),(padleft,padright),(0,0))

def check_coord(coord, size=200):
    """
        check coordination
    """
    xmin, ymin, xmax, ymax = coord
    if (xmax - xmin) > size:
        xc = (xmin+xmax)//2
        xmin, xmax = xc-(size//2), xc+(size//2)
    if (ymax - ymin) > size:
        yc = (ymin+ymax)//2
        ymin, ymax = yc-(size//2), yc+(size//2)
    return xmin, ymin, xmax, ymax

def crop_adjusted_std(img, coord, delta=100):
    """
        adjusting crop and padding constant and std
    """
    coord = check_coord(coord)
    xmin, ymin, xmax, ymax = coord
    img = (img - np.mean(img))/np.std(img)*32+128
    img = img[ymin:ymax, xmin:xmax].copy()
    padding = compute_padding(coord)
    img = np.pad(img, padding, "constant")
    return img[None,:]

def load_semantic_vector(path):
    """
        load semantic vector
        - path <str>
    """
    with open(path, "r") as f:
        lines = f.readlines()
    semantic_vectors = []
    for vec in lines:
        semantic_vectors.append(np.asarray(vec.split("\n")[0].split(" ")).astype("float32"))
    return torch.from_numpy(np.asarray(semantic_vectors))