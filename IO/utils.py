import os
from os import listdir as ld
from os.path import join as pj
from PIL import Image
import numpy as np
import torch


def save_images_without_duplication(data_root, anno_folders, save_image_folder):
    """
        unused
    """
    if os.path.exists(save_image_folder) is False:
        os.makedirs(save_image_folder)
        annos, imgs = load_path(data_root, anno_folders)
        images = load_images(imgs)
        annotations_path = load_annotations_path(annos, images)
        images_path = load_images_path(imgs, annotations_path)
        for k,v in images_path.items():
            Image.fromarray(images[k].astype(np.uint8)).save(pj(save_image_folder,k+".png"))
    else:
        print("folder is already exists")

        
def make_imagelist(image_data_root, imagelist_path):
    """
        unused
    """
    image_list = ld(image_data_root)
    if ".ipynb_checkpoints" in image_list:
        image_list.remove(".ipynb_checkpoints")
    
    image_list = [pj(image_data_root,filename) for filename in image_list]
    with open(imagelist_path, "w") as f:
        for image_path in image_list:
            f.write(image_path+"\n")

            
def format_output(coords, fid, width=4608, height=2592):
    header = r"""<annotation>
        <folder>images</folder>
        <filename>{0}.JPG</filename>
        <path>C:\Users\ooe\Desktop\labelImg-master\images\{0}.JPG</path>
        <source>
            <database>Unknown</database>
        </source>
        <size>
            <width>{1}</width>
            <height>{2}</height>
            <depth>3</depth>
        </size>
        <segmented>0</segmented>
    """
    
    obj_box = """
        <object>
            <name>{}</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>{}</xmin>
                <ymin>{}</ymin>
                <xmax>{}</xmax>
                <ymax>{}</ymax>
            </bndbox>
        </object>
    """
    
    footer = """
    </annotation>
    """
    
    content = header.format(fid, width, height)
    for name, (xmin, ymin, xmax, ymax) in coords:
        bbox = obj_box.format(name, xmin, ymin, xmax, ymax)
        content = content + bbox
    content = content + footer
    return content


def output_formatter(result, thresh=0.3):
    """
        formatting result to labelImg XML style
        - result: {file id: np.asarray([x1, y1, x2, y2, probability], ...)}
        - thresh: float
    """
    output = {}
    for fid, elements in result.items():
        coords = []
        for element in elements:
            if element[-1] > thresh:
                coords.append([int(point) for point in element[:-1]])
        output.update({fid: np.asarray([("insects", coord) for coord in coords])})
    return output


def write_output_xml(outputs, path):
    """
        write labelImg XML using outputs
        - outputs: {file id: np.asarray(['insects', [x1, y1, x2, y2]], ...)}
        - path: str
    """
    if os.path.exists(path) is False:
        os.makedirs(path)
        for fid, coords in outputs.items():
            content = format_output(coords, fid)
            fp = pj(path, fid + ".xml")
            with open(fp, "w") as f:
                f.write(content)
    else:
        print("folder is already exist. check and move folder.")
        

def refine_result_by_ovthresh(result, ovthresh=0.3):
    """
        refine result by ovthresh
        - result: {image_id: {label_id: np.asarray([[x1, y1, x2, y2, conf], ...])}}
        - ovthresh: float
    """
    conf_refined_result = {}
    for image_id, result_per_label in result.items():
        conf_refined_result_per_label = {}
        for label_id, res in result_per_label.items():
            refined_result_per_res = []
            for box in res:
                if box[4] > ovthresh:
                    refined_result_per_res.append(box.tolist())
            conf_refined_result_per_label.update({label_id: np.asarray(refined_result_per_res)})
        conf_refined_result.update({image_id: conf_refined_result_per_label})
    return conf_refined_result