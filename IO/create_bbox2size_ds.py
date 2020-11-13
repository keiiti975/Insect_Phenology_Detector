import numpy as np
import pandas as pd
import os
from os import getcwd as cwd
from os.path import join as pj
from IO.build_ds import load_anno, create_annotation

"""
use this script in working directory
==> Insect_Phenology_Detector/
"""

def divide_target_and_body(new_anno):
    """
        divide body size from new_anno, {image_id: {"target", "body"}}
        Args:
            - new_anno: {image_id: list(tuple(insect_name, coord))}
    """
    new_anno_div_body = {}
    for image_id, values in new_anno.items():
        target_list = []
        body_list = []
        for value in values:
            if value[0] == 'body size':
                body_list.append(value)
            else:
                target_list.append(value)
        new_anno_div_body.update({image_id: {"target": target_list, "body": body_list}})
    return new_anno_div_body


def get_feature_point_filter(target_bbox, body_bboxes):
    """
        filtering features of the head and tail
        Args:
            - target_bbox: np.array(dtype=int), shape==[4]
            - body_bboxes: np.array(dtype=int), shape==[body_num, 4]
    """
    x1_filter = body_bboxes[:, 0] >= target_bbox[0]
    x2_filter = body_bboxes[:, 2] <= target_bbox[2]
    y1_filter = body_bboxes[:, 1] >= target_bbox[1]
    y2_filter = body_bboxes[:, 3] <= target_bbox[3]
    feature_point_filter = x1_filter & x2_filter & y1_filter & y2_filter
    return feature_point_filter


def calc_euclidean_distance(body_bboxes_filtered_feature_point):
    """
        calculate euclidean distance between head and tail
        Args:
            - body_bboxes_filtered_feature_point: np.array(dtype=int), shape==[2, 4]
    """
    p1 = np.array([
        (body_bboxes_filtered_feature_point[0, 2] + body_bboxes_filtered_feature_point[0, 0]) / 2, 
        (body_bboxes_filtered_feature_point[0, 3] + body_bboxes_filtered_feature_point[0, 1]) / 2, 
    ])
    p2 = np.array([
        (body_bboxes_filtered_feature_point[1, 2] + body_bboxes_filtered_feature_point[1, 0]) / 2, 
        (body_bboxes_filtered_feature_point[1, 3] + body_bboxes_filtered_feature_point[1, 1]) / 2, 
    ])
    return np.linalg.norm(p1 - p2)


def get_new_anno_with_size(new_anno_div_body):
    """
        create new_anno with insect size, {image_id: list(tuple(insect_name, coord))}
        Args:
            - new_anno_div_body: {image_id: {
                "target": list(tuple(insect_name, coord)), 
                "body": list(tuple(insect_name, coord))
                }}
    """
    new_anno_with_size = {}
    for image_id, values in new_anno_div_body.items():
        target_list = values['target']
        body_list = values['body']
        body_bboxes = np.array([elem_body[1] for elem_body in body_list])
        target_list_with_size = []
        for elem_target in target_list:
            target_bbox = np.array(elem_target[1])
            feature_point_filter = get_feature_point_filter(target_bbox, body_bboxes)
            if feature_point_filter.sum() == 2:
                body_bboxes_filtered_feature_point = body_bboxes[feature_point_filter]
                distance = calc_euclidean_distance(body_bboxes_filtered_feature_point)
                elem_target_with_size = list(elem_target)
                elem_target_with_size.append(distance)
                elem_target_with_size = tuple(elem_target_with_size)
                target_list_with_size.append(elem_target_with_size)
        new_anno_with_size.update({image_id: target_list_with_size})
    return new_anno_with_size


def get_bbox_df(new_anno_with_size):
    """
        create bbox df, pd.DataFrame({"width", "height", "label", "size"})
        Args:
            - new_anno_with_size: {image_id: list(tuple(insect_name, coord, size))}
    """
    # create array
    width_array = []
    height_array = []
    label_array = []
    size_array = []
    for image_id, values in new_anno_with_size.items():
        for value in values:
            width_array.append(value[1][2] - value[1][0])
            height_array.append(value[1][3] - value[1][1])
            label_array.append(value[0])
            size_array.append(value[2])
    width_array = np.array(width_array)
    height_array = np.array(height_array)
    label_array = np.array(label_array)
    size_array = np.array(size_array)
    
    # convert insect_name to label
    idx = np.unique(label_array)
    name_to_lbl = {}
    for i, elem_idx in enumerate(idx):
        name_to_lbl.update({elem_idx: i})
    label_array = np.array([name_to_lbl[elem_label_array] for elem_label_array in label_array])
    print(name_to_lbl)
    
    return pd.DataFrame({"width": width_array, 
                         "height": height_array, 
                         "label": label_array, 
                         "size": size_array})


if __name__ == "__main__":
    data_root = pj(cwd(), "data")
    bbox_data_path = pj(cwd(), "data/bbox_data", "target_only_20200806.csv")
    img_folder = "refined_images"
    anno_folders = ["annotations_0", "annotations_2", "annotations_3", "annotations_4", "annotations_20200806"]
    
    unused_labels = [']', 'Coleoptera', 'Hemiptera', 
                     'Hymenoptera', 'Megaloptera', 'Unknown', 
                     'unknown', 'medium insect', 'small insect', 
                     'snail', 'spider']
    images, anno = load_anno(data_root, img_folder, anno_folders, return_body=True)
    new_anno = create_annotation(images, anno, unused_labels, False, False)
    new_anno_div_body = divide_target_and_body(new_anno)
    new_anno_with_size = get_new_anno_with_size(new_anno_div_body)
    bbox_df = get_bbox_df(new_anno_with_size)
    if os.path.exists(os.path.dirname(bbox_data_path)) is False:
        os.makedirs(os.path.dirname(bbox_data_path))
    bbox_df.to_csv(bbox_data_path)
    print("create data to " + bbox_data_path)