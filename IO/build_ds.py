import numpy as np
import os
from os.path import join as pj
import pandas as pd
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from IO.loader import load_path, load_images, load_annotations_path, load_annotations, get_label_dic
from utils.crop import crop_adjusted_std, crop_adjusted_std_resize


def build_classification_ds(anno, images, crop, size=200, return_sizes=False):
    """
        build classification dataset
        - anno: {image_id: list(tuple(insect_name, coord))}
        - images: {image_id: image}
        - crop: choice from [crop_standard, crop_adjusted, crop_adjusted_std, crop_adjusted_std_resize]
        - size: image size after crop and padding
        - return_sizes: if true, return insect sizes
            this parameter need anno: {image_id: list(tuple(insect_name, coord, insect_size))}
    """
    if return_sizes is True:
        imgs, lbls, sizes = [], [], []
        for k, v in tqdm(anno.items()):
            img = images[k]
            for lbl, coord, size in v:
                xmin, ymin, xmax, ymax = coord
                if (xmin != xmax) and (ymin != ymax):
                    x = crop(img, coord, size//2)
                    imgs.append(x)
                    lbls.append(lbl)
                    sizes.append(size)
                else:
                    continue
        imgs = np.concatenate(imgs)
        sizes = np.asarray(sizes)
        lbl_dic = {k:i for i,k in enumerate(np.unique(lbls))}
        print(lbl_dic)
        lbls = np.asarray(list(map(lambda x:lbl_dic[x], lbls)))
        return imgs.astype("int32"), lbls, sizes.astype("float")
    else:
        imgs, lbls = [], []
        for k, v in tqdm(anno.items()):
            img = images[k]
            for lbl, coord in v:
                xmin, ymin, xmax, ymax = coord
                if (xmin != xmax) and (ymin != ymax):
                    x = crop(img, coord, size//2)
                    imgs.append(x)
                    lbls.append(lbl)
                else:
                    continue
        imgs = np.concatenate(imgs)
        lbl_dic = {k:i for i,k in enumerate(np.unique(lbls))}
        print(lbl_dic)
        lbls = np.asarray(list(map(lambda x:lbl_dic[x], lbls)))
        return imgs.astype("int32"), lbls
    

def load_anno(data_root, img_folder, anno_folders, return_body=False):
    """
        load anno
        Args:
            - data_root: str
            - img_folder: str
            - anno_folders: list(dtype=str)
    """
    print("loading path ...")
    annos, imgs = load_path(data_root, img_folder, anno_folders)
    print("loading images ...")
    images = load_images(imgs)
    annotations_path = load_annotations_path(annos, images)
    print("loading annos ...")
    anno = load_annotations(annotations_path, return_body)
    return images, anno


def load_label_dic(anno, each_flag=False, plus_other=False, target_with_other=False):
    """
        load label_dic with experiment setting
        Args:
            - anno: {image_id: list(tuple(insect_name, coord))}
            - each_flag: bool
            - plus_other: bool, if divide into target_class + other_class
            - target_with_other: bool, if divide into target_class(split label) + other_class
    """
    if plus_other is True:
        label_dic = {'Coleoptera': 1,
                     'Diptera': 0,
                     'Ephemeridae': 0,
                     'Ephemeroptera': 0,
                     'Hemiptera': 1,
                     'Hymenoptera': 1,
                     'Lepidoptera': 0,
                     'Megaloptera': 1,
                     'Plecoptera': 0,
                     'Trichoptera': 0,
                     'medium insect': 1,
                     'small insect': 1,
                     'snail': 1,
                     'spider': 1,
                     'Unknown': 1}
    elif target_with_other is True:
        label_dic = {'Coleoptera': 6, 
                    'Diptera': 0, 
                    'Ephemeridae': 1, 
                    'Ephemeroptera': 2, 
                    'Hemiptera': 6, 
                    'Hymenoptera': 6, 
                    'Lepidoptera': 3, 
                    'Megaloptera': 6, 
                    'Plecoptera': 4, 
                    'Trichoptera': 5, 
                    'medium insect': 6, 
                    'small insect': 6, 
                    'snail': 6, 
                    'spider': 6,
                    'Unknown': 6}
    else:
        label_dic = get_label_dic(anno, each_flag=each_flag, make_refinedet_data=True)
    print(label_dic)
    return label_dic


def get_coord(value, centering=False):
    """
        translate value to (x_center, y_center, w, h) or (x1, y1, x2, y2)
        Args:
            - value: list(dtype=int), [x1, y1, x2, y2]
            - centering: bool
    """
    if centering is True:
        width = (value[2] - value[0])/2
        height = (value[3] - value[1])/2
        x_center = value[0] + width
        y_center = value[1] + height
        return x_center, y_center, width, height
    else:
        upper_left_x = value[0]
        upper_left_y = value[1]
        under_right_x = value[2]
        under_right_y = value[3]
        return upper_left_x, upper_left_y, under_right_x, under_right_y
    
    
def coord_in_percentage_notation(coord, image_size, centering=False):
    """
        get value in percentage notation
        Args:
            - value: list(dtype=int), [x1, y1, x2, y2]
            - image_size: tuple(dtype=int), [height, width]
            - centering: bool
    """
    if centering is True:
        coord[0] /= 2*image_size[1]
        coord[1] /= 2*image_size[0]
        coord[2] /= image_size[1]
        coord[3] /= image_size[0]
    else:
        coord[0] /= image_size[1]
        coord[1] /= image_size[0]
        coord[2] /= image_size[1]
        coord[3] /= image_size[0]
    return coord


def create_annotation(images, anno, unused_labels, centering=False, percent=False):
    """
        adopt coord to training condition
        Args:
            - images: {image_id: image}
            - anno: {image_id: list(tuple(insect_name, coord))}
            - unused_labels: list(dtype=str)
            - centering: bool
            - percent: bool
    """
    new_anno = {}
    for k,v in tqdm(anno.items()):
        new_value = []
        if k == ".ipynb_checkpoints":
            continue
        image_size = images[k].shape[0:2]
        for value in anno[k]:
            if value[0] in unused_labels:
                continue
            coord = get_coord(value[1], centering)
            if percent is True:
                coord = coord_in_percentage_notation(list(coord), image_size, centering)
            if value[0] in ["unknown", "medium insect", "small insect"]:
                label = "Unknown"
            else:
                label = value[0]
            new_annotation = (label, coord)
            new_value.append(new_annotation)
        new_anno.update({k:new_value})
    return new_anno


def init_data_dic(label_dic, new_anno, new_anno_not_percent=None):
    """
        initialize data_dic
        Args:
            - label_dic: {insect_name, label}
            - new_anno: {image_id: list(tuple(insect_name, coord))}
                coord in percentage notation
            - new_anno_not_percent: {image_id: list(tuple(insect_name, coord))}
                coord in pixel notation, use if result coord in pixel notation
    """
    data_dic = {}
    for k, v in label_dic.items():
        data_dic.update({k: {"size": [], "file_id": [], "coord": []}})

    for k, v in new_anno.items():
        if k == ".ipynb_checkpoints":
            continue
        for i, value in enumerate(v):
            data_dic[value[0]]["size"].append([value[1][2] - value[1][0], value[1][3] - value[1][1]])
            data_dic[value[0]]["file_id"].append(k)
            if new_anno_not_percent is not None:
                data_dic[value[0]]["coord"].append(new_anno_not_percent[k][i][1])
            else:
                data_dic[value[0]]["coord"].append(value[1])
            
    for k, v in data_dic.items():
        data_dic[k]["size"] = np.array(data_dic[k]["size"])
        data_dic[k]["file_id"] = np.array(data_dic[k]["file_id"])
        data_dic[k]["coord"] = np.array(data_dic[k]["coord"])
    
    return data_dic


def get_dbscan_result(X_data, eps=0.005, min_samples=5):
    """
        get DBSCAN clastering result
        Args:
            - X_data: pd.DataFrame
            - eps: float
            - min_samples: int
    """
    leaf_size = 30
    n_jobs = 1

    db = DBSCAN(eps=eps, min_samples=min_samples, leaf_size=leaf_size, n_jobs=n_jobs)
    
    X_data_dbscanClustered = db.fit_predict(X_data.loc[:, :])
    cluster_sum_X_data_dbscanClustered = []
    for value in X_data_dbscanClustered:
        if value > 0:
            cluster_sum_X_data_dbscanClustered.append(0)
        else:
            cluster_sum_X_data_dbscanClustered.append(value)
    cluster_sum_X_data_dbscanClustered = pd.DataFrame(data=cluster_sum_X_data_dbscanClustered, index=X_data.index, columns=['cluster'])
    return cluster_sum_X_data_dbscanClustered


def adopt_DBSCAN_to_data_dic(data_dic):
    """
        adopt DBSCAN to data_dic
        Args:
            - data_dic: {insect_name: {
                "size": list(dtype=float), 
                "file_id": list(dtype=float), 
                "coord": list(dtype=float)
            }}
    """
    for k, v in data_dic.items():
        data_index = range(0, len(v["size"]))
        X_data = pd.DataFrame(data=v["size"], index=data_index)

        X_data_dbscanClustered = get_dbscan_result(X_data)

        abnormal_id = []
        for i, val in enumerate(X_data_dbscanClustered.cluster):
            if val != 0:
                abnormal_id.append(i)
        abnormal_filter = np.ones(len(X_data), dtype="bool")
        abnormal_filter[abnormal_id] = False

        data_dic[k]["size"] = data_dic[k]["size"][abnormal_filter]
        data_dic[k]["file_id"] = data_dic[k]["file_id"][abnormal_filter]
        data_dic[k]["coord"] = data_dic[k]["coord"][abnormal_filter]
    
    return data_dic


def create_DBSCAN_filtered_annotation(new_anno, data_dic):
    """
        create DBSCAN filtered annotation from data_dic
        Args:
            - new_anno: {image_id: list(tuple(insect_name, coord))}
            - data_dic: {insect_name: {
                "size": list(dtype=float), 
                "file_id": list(dtype=float), 
                "coord": list(dtype=float)
            }}
    """
    DBSCAN_filtered_new_anno = {}
    for k, v in new_anno.items():
        DBSCAN_filtered_new_anno.update({k: []})

    for anno_k, anno_v in new_anno.items():
        for data_k, data_v in data_dic.items():
            file_id_filter = data_v["file_id"] == anno_k
            file_id_filtered_coord = data_v["coord"][file_id_filter]
            new_annotation = [(data_k, coord.tolist()) for coord in file_id_filtered_coord]
            DBSCAN_filtered_new_anno[anno_k].extend(new_annotation)
    
    return DBSCAN_filtered_new_anno


def adopt_DBSCAN(label_dic, new_anno, new_anno_not_percent=None):
    """
        adopt DBSCAN to new_anno
        Args:
            - label_dic: {insect_name, label} 
            - new_anno: {image_id: list(tuple(insect_name, coord))}
                coord is percentage notation
            - new_anno_not_percent: {image_id: list(tuple(insect_name, coord))}
                coord is pixel notation, use if result coord in pixel notation
    """
    data_dic = init_data_dic(label_dic, new_anno, new_anno_not_percent)
    data_dic = adopt_DBSCAN_to_data_dic(data_dic)
    DBSCAN_filtered_new_anno = create_DBSCAN_filtered_annotation(new_anno, data_dic)
    return DBSCAN_filtered_new_anno


def write_annotation(new_anno, label_dic, label_path, last_flag=False):
    """
        output annotation file
        Args:
            - new_anno: {image_id: list(tuple(insect_name, coord))}
            - label_dic: {insect_name, label}
            - label_path: str
            - last_flag: bool
    """
    labels = []
    os.makedirs(label_path)
    for k,v in tqdm(new_anno.items()):
        path = pj(label_path, k+".txt")
        with open(path, "w") as f:
            for value in new_anno[k]:
                label = label_dic[value[0]]
                labels.append(label)
                coord = value[1]
                if last_flag is False:
                    line = str(label)+" "+str(coord[0])+" "+str(coord[1])+" "+str(coord[2])+" "+str(coord[3])+"\n"
                else:
                    line = str(coord[0])+" "+str(coord[1])+" "+str(coord[2])+" "+str(coord[3])+" "+str(label)+"\n"
                f.write(line)
    labels = np.array(labels)
    idx, count = np.unique(labels, return_counts=True)
    print("idx = {}".format(idx))
    print("count = {}".format(count))
    
    
def build_detection_dataset_as_txt(data_root, img_folder, anno_folders, label_path, 
                                  each_flag=False, plus_other=False, target_with_other=False, 
                                  centering=False, percent=False, last_flag=False, 
                                   use_DBSCAN=False):
    """
        build detection dataset as txt
        Args:
            - data_root: str
            - img_folder: str
            - anno_folders: list(dtype=str)
            - label_path: str
            - each_flag: bool
            - plus_other: bool, if divide into target_class + other_class
            - target_with_other: bool, if divide into target_class(split label) + other_class
            - centering: bool
            - percent: bool
            - last_flag: bool
            - use_DBSCAN: bool
    """
    unused_labels = ["]"]
    images, anno = load_anno(data_root, img_folder, anno_folders)
    new_anno = create_annotation(images, anno, unused_labels, centering, percent)
    label_dic = load_label_dic(new_anno, each_flag, plus_other, target_with_other)
    if use_DBSCAN is True:
        new_anno = adopt_DBSCAN(label_dic, new_anno)
    write_annotation(new_anno, label_dic, label_path, last_flag)


def build_classification_ds_from_result(images, result, use_resize=False):
    """
        make classification dataset using result coord
        - images: {file id: <np.array>}
        - result: {file id: {label id: <np.array>}}
        - use_resize: bool
    """
    insect_dataset = {}
    for image_id, res in result.items():
        print("creating images: {}".format(image_id))
        res = res[0]
        default_imgs = images[image_id]
        cropped_imgs = []
        for box in tqdm(res):
            coord = box[:4]
            if use_resize is True:
                cropped_img = crop_adjusted_std_resize(default_imgs, coord, 100, use_integer_coord=True)
            else:
                cropped_img = crop_adjusted_std(default_imgs, coord, 100, use_integer_coord=True)
            cropped_imgs.append(cropped_img)
        cropped_imgs = np.concatenate(cropped_imgs)
        insect_dataset.update({image_id: cropped_imgs.astype("float32")})
    return insect_dataset