import numpy as np
import os
from os.path import join as pj
from tqdm import tqdm
from IO.loader import load_path, load_images, load_annotations_path, load_annotations, get_label_dic
from utils.crop import crop_adjusted_std, crop_adjusted_std_resize


def build_detection_dataset(anno, images, anno_func, std=True):
    """
        unused
    """
    targets = []
    annotated_images = []
    for k,img in tqdm(images.items()):
        if k in anno:
            coords = map(lambda x:x[1], anno[k])
            if std is True:
                img = (img - np.mean(img))/np.std(img)*32+128
            target = make_detect_target(img, coords, anno_func)
            targets.append(target[None,:])
            annotated_images.append(img[None,:])
    annotated_images = pad_full_images(annotated_images)
    targets = pad_full_images(targets)
    return annotated_images, targets

def build_classification_ds(anno, images, crop, size=200, return_sizes=False):
    """
        build classification dataset
        - anno: {file id: [(insect name, [x1, y1, x2, y2]), ...]}
        - images: {file id: image data, np.array}
        - crop: choice from [crop_standard, crop_adjusted, crop_adjusted_std, crop_adjusted_std_resize]
        - size: image size after crop and padding
        - return_sizes: if true, return sizes
    """
    imgs, lbls, sizes = [],[],[]
    for k,v in tqdm(anno.items()):
        img = images[k]
        for lbl, coord in v:
            xmin, ymin, xmax, ymax = coord
            if (xmin != xmax) and (ymin != ymax):
                x = crop(img, coord, size//2)
                imgs.append(x)
                lbls.append(lbl)
                sizes.append((xmax - xmin) * (ymax - ymin))
            else:
                continue
    imgs = np.concatenate(imgs)
    sizes = np.asarray(sizes)
    lbl_dic = {k:i for i,k in enumerate(np.unique(lbls))}
    print(lbl_dic)
    lbls = np.asarray(list(map(lambda x:lbl_dic[x], lbls)))
    if return_sizes is True:
        return imgs.astype("int32"), lbls, sizes
    else:
        return imgs.astype("int32"), lbls


def build_detection_dataset_as_txt(data_root, img_folder, anno_folders, label_path, each_flag=False, last_flag=False, centering=False, percent=False, check_label_folder=True):
    """
        build detection dataset, pascal voc style
        - data_root: str
        - img_folder: str
        - anno_folders: [str, ...]
        - label_path: str
        - each_flag: bool
        - last_flag: bool
        - centering: bool
        - percent: bool
        - check_label_folder: bool
    """
    print("loading path ...")
    annos, imgs = load_path(data_root, img_folder, anno_folders)
    print("loading images ...")
    images = load_images(imgs)
    annotations_path = load_annotations_path(annos, images)
    print("loading annos ...")
    anno = load_annotations(annotations_path)
    label_dic = get_label_dic(anno, each_flag=each_flag, make_refinedet_data=True)

    if check_label_folder is False or os.path.exists(label_path) is False:
        if check_label_folder is True:
            os.makedirs(label_path)
        if centering is True:
            new_anno = {}
            for k,v in tqdm(anno.items()):
                new_value = []
                if k == ".ipynb_checkpoints":
                    continue
                for value in anno[k]:
                    if percent is False:
                        width = (value[1][2] - value[1][0])/2
                        height = (value[1][3] - value[1][1])/2
                        x_center = value[1][0] + width
                        y_center = value[1][1] + height
                    else:
                        image_size = images[k].shape[0:2]
                        width = (value[1][2] - value[1][0])/(2*image_size[1])
                        height = (value[1][3] - value[1][1])/(2*image_size[0])
                        x_center = (value[1][0] + width)/image_size[1]
                        y_center = (value[1][1] + height)/image_size[0]
                    new_coord = [x_center, y_center, width, height]
                    new_annotation = (value[0], new_coord)
                    new_value.append(new_annotation)
                new_anno.update({k:new_value})

            for k,v in tqdm(new_anno.items()):
                path = pj(label_path, k+".txt")
                with open(path, "w") as f:
                    for value in new_anno[k]:
                        label = label_dic[value[0]]
                        coord = value[1]
                        if last_flag is False:
                            line = str(label)+" "+str(coord[0])+" "+str(coord[1])+" "+str(coord[2])+" "+str(coord[3])+"\n"
                        else:
                            line = str(coord[0])+" "+str(coord[1])+" "+str(coord[2])+" "+str(coord[3])+" "+str(label)+"\n"
                        f.write(line)
        else:
            new_anno = {}
            for k,v in tqdm(anno.items()):
                new_value = []
                if k == ".ipynb_checkpoints":
                    continue
                for value in anno[k]:
                    if percent is False:
                        upper_left_x = value[1][0]
                        upper_left_y = value[1][1]
                        under_right_x = value[1][2]
                        under_right_y = value[1][3]
                    else:
                        image_size = images[k].shape[0:2]
                        upper_left_x = value[1][0]/image_size[1]
                        upper_left_y = value[1][1]/image_size[0]
                        under_right_x = value[1][2]/image_size[1]
                        under_right_y = value[1][3]/image_size[0]
                    new_coord = [upper_left_x, upper_left_y, under_right_x, under_right_y]
                    new_annotation = (value[0], new_coord)
                    new_value.append(new_annotation)
                new_anno.update({k:new_value})

            for k,v in tqdm(new_anno.items()):
                path = pj(label_path, k+".txt")
                with open(path, "w") as f:
                    for value in new_anno[k]:
                        label = label_dic[value[0]]
                        coord = value[1]
                        if last_flag is False:
                            line = str(label)+" "+str(coord[0])+" "+str(coord[1])+" "+str(coord[2])+" "+str(coord[3])+"\n"
                        else:
                            line = str(coord[0])+" "+str(coord[1])+" "+str(coord[2])+" "+str(coord[3])+" "+str(label)+"\n"
                        f.write(line)
    else:
        print("folder is already exists")
        

def build_classification_ds_from_result(images, result, use_resize=True):
    """
        make classification dataset using result coord
        - images: {file id: image data, np.array}
        - result: {file id: {'coord'}}
        - use_resize: bool
    """
    insect_dataset = {}
    for image_id, res in result.items():
        default_img = images[image_id]
        imgs = []
        for box in tqdm(res['coord'], leave=False):
            coord = box[:4]
            if use_resize is True:
                img = crop_adjusted_std_resize(default_img, coord, 100, use_integer_coord=True)
            else:
                img = crop_adjusted_std(default_img, coord, 100, use_integer_coord=True)
            imgs.append(img)
        imgs = np.concatenate(imgs)
        imgs = imgs.transpose(0, 3, 1, 2)
        insect_dataset.update({image_id: imgs.astype("float32")})
    return insect_dataset


def pad_full_image(img, target_shape):
    """
        unused
    """
    shape  = img.shape
    padleft    = (target_shape[1] - shape[1]) // 2
    padright   = target_shape[1] - padleft - shape[1]
    padtop     = (target_shape[2] - shape[2]) // 2
    padbottom  = target_shape[2] - padtop - shape[2]
    if len(shape)==4:
        padding = ((0,0), (padleft,padright), (padtop,padbottom), (0,0))
    else:
        padding = ((0,0), (padleft,padright), (padtop,padbottom))
    return np.pad(img, padding, "constant")


def pad_full_images(images):
    """
        unused
    """
    shape = [im.shape for im in images]
    shape = np.max(shape, 0)
    return np.concatenate([pad_full_image(img, shape) for img in images])