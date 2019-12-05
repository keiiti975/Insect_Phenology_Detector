import cv2
import numpy as np
from os import listdir as ld
from os.path import join as pj
from PIL import Image
from scipy import sparse
import torch
from tqdm import tqdm
import xml.etree.ElementTree as ET


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
    x = filter(lambda x: x.tag == "object", tree.getroot())
    return filter(lambda x: get_name(x) != "body size", x)


def parse_annotations(fp):
    """
        XML annotation file kara
        annotation list shutsuryoku
    """
    return [(get_name(objects), get_coords(objects)) for objects in get_objects(fp)]


def file_id(anno):
    """
        get filename from annotation path
    """
    return anno.split("/")[-1][:13]


def load_path(root="/home/tanida/workspace/insects_project/data", anno_folders=["annotations_0"]):
    """
        load pathes
    """
    annos = []
    for anno_folder in anno_folders:
        annos_name = ld(pj(root, anno_folder))
        annos.extend([pj(root, anno_folder, x) for x in annos_name])
    imgs = ld(pj(root, "refined_images"))
    imgs = [pj(root, "refined_images", x) for x in imgs]
    return annos, imgs


def load_images(imgs):
    """
        load images and map to filename
    """
    images = {file_id(im): np.array(Image.open(im)) for im in imgs}
    return images


def load_annotations_path(annos, images):
    """
        load annotations path and map to filename
    """
    annotations_path = {
        idx: list(filter(lambda x: idx in x, annos)) for idx in images}
    annotations_path = {k: v for k,
                        v in annotations_path.items() if len(v) > 0}
    return annotations_path


def load_images_path(imgs, annotations_path):
    """
        load images path and map to filename
    """
    images_path = {idx: list(filter(lambda x: idx in x, imgs))[
        0] for idx in annotations_path}
    return images_path


def load_annotations(annotations_path):
    """
        load annotations
    """
    anno = {}
    for k, v in annotations_path.items():
        anno[k] = []
        for x in filter(lambda x: x.endswith(".xml"), v):
            anno[k].extend(parse_annotations(x))
    return anno


def get_all_label_dic(anno):
    """
        get all_label_dic from annotations
    """
    lbl = []
    for k, v in anno.items():
        for value in v:
            lbl.append(value[0])
    lbl = np.asarray(lbl)
    lbls, counts = np.unique(lbl, return_counts=True)
    label_dic = {k: 1 for k in lbls}
    return label_dic


def get_each_label_dic(anno):
    """
        get each_label_dic from annotations
    """
    lbl = []
    for k, v in anno.items():
        for value in v:
            lbl.append(value[0])
    lbl = np.asarray(lbl)
    lbls, counts = np.unique(lbl, return_counts=True)
    label_dic = {k: i for i, k in enumerate(lbls)}
    return label_dic


def rank_roidb_ratio(roidb):
    """
        rank roidb based on the ratio between width and height.
    """
    ratio_large = 2  # largest ratio to preserve.
    ratio_small = 0.5  # smallest ratio to preserve.

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
    for k, v in anno.items():
        boxes = {"boxes": np.array([value[1]
                                    for value in anno[k]], dtype="uint16")}
        names = [value[0] for value in anno[k]]
        gt_classes = {"gt_classes": np.array(
            [label_dic[name] for name in names], dtype="int32")}
        overlap = [[1 if i == idx else 0 for i in range(
            10)] for idx in gt_classes["gt_classes"]]
        gt_overlap = {"gt_overlap": sparse.csr_matrix(
            np.array(overlap, dtype="float32"))}
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


def get_all_anno_recs(anno):
    """
        get all annotation recs
    """
    imagenames = []
    recs = {}
    for k, v in anno.items():
        imagenames.append(k)
        rec_image = []
        for i in range(len(v)):
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


def get_each_anno_recs(anno):
    """
        get each annotation recs
    """
    imagenames = []
    recs = {}
    for k, v in anno.items():
        imagenames.append(k)
        rec_image = []
        for i in range(len(v)):
            name = {"name": v[i][0]}
            difficult = {"difficult": 0}
            bbox = {"bbox": v[i][1]}

            rec = {}
            rec.update(name)
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
    padleft = (2 * delta - (xmax - xmin)) // 2
    padright = 2 * delta - padleft - (xmax - xmin)
    padtop = (2 * delta - (ymax - ymin)) // 2
    padbottom = 2 * delta - padtop - (ymax - ymin)
    return ((padtop, padbottom), (padleft, padright), (0, 0))


def check_coord(coord, size=200):
    """
        check coordination
    """
    xmin, ymin, xmax, ymax = coord
    if (xmax - xmin) > size:
        xc = (xmin + xmax) // 2
        xmin, xmax = xc - (size // 2), xc + (size // 2)
    if (ymax - ymin) > size:
        yc = (ymin + ymax) // 2
        ymin, ymax = yc - (size // 2), yc + (size // 2)
    return int(xmin), int(ymin), int(xmax), int(ymax)


def crop_adjusted_std(img, coord, delta=100):
    """
        adjusting crop and padding constant and std
    """
    coord = check_coord(coord)
    xmin, ymin, xmax, ymax = coord
    img = (img - np.mean(img, keepdims=True)) / \
        np.std(img, keepdims=True) * 32 + 128
    img = img[ymin:ymax, xmin:xmax].copy()
    padding = compute_padding(coord)
    img = np.pad(img, padding, "constant")
    return img[None, :]


def crop_adjusted_std_resize(img, coord, delta=100):
    """
        adjusting crop and padding constant and std
    """
    coord = check_coord(coord)
    xmin, ymin, xmax, ymax = coord
    if (xmax - xmin) > (ymax - ymin):
        max_length_axis = xmax - xmin
    else:
        max_length_axis = ymax - ymin
    img = (img - np.mean(img, keepdims=True)) / \
        np.std(img, keepdims=True) * 32 + 128
    img = img[ymin:ymax, xmin:xmax].copy()
    if (xmax - xmin) > (ymax - ymin):
        img = cv2.resize(img, dsize=(200, (int)(
            img.shape[0] * 200 / max_length_axis)), interpolation=cv2.INTER_LINEAR)
    else:
        img = cv2.resize(img, dsize=((int)(
            img.shape[1] * 200 / max_length_axis), 200), interpolation=cv2.INTER_LINEAR)
    padding = compute_padding((0, 0, img.shape[1], img.shape[0]))
    img = np.pad(img, padding, "constant")
    return img[None, :]


def refine_result_by_ovthresh(result, ovthresh=0.3):
    conf_refined_result = {}
    for image_id, res in result.items():
        refined_result_per_res = []
        for box in res['coord']:
            if box[4] > ovthresh:
                refined_result_per_res.append(box.tolist())
        conf_refined_result.update(
            {image_id: {"coord": np.asarray(refined_result_per_res)}})
    return conf_refined_result


def build_classification_ds(images, result, use_resize=True):
    insect_dataset = {}
    for image_id, res in result.items():
        default_img = images[image_id]
        imgs = []
        for box in tqdm(res['coord'], leave=False):
            coord = box[:4]
            if use_resize is True:
                img = crop_adjusted_std_resize(default_img, coord, 100)
            else:
                img = crop_adjusted_std(default_img, coord, 100)
            imgs.append(img)
        imgs = np.concatenate(imgs)
        imgs = imgs.transpose(0, 3, 1, 2)
        insect_dataset.update({image_id: imgs.astype("float32")})
    return insect_dataset