import cv2
import numpy as np
from os import listdir as ld
from os.path import join as pj
from PIL import Image
import torch
import torch.utils.data as data


class insects_dataset_from_voc_style_txt(data.Dataset):
    """
        class for creating voc_stle insects_dataset
    """

    def __init__(self, image_root, resize_size, crop_num, model_name, training=False, target_root=None):
        self.image_root = image_root
        self.resize_size = resize_size
        self.crop_num = crop_num
        self.model_name = model_name
        self.training = training
        self.target_root = target_root
        self.ids = self.get_ids()

    def __getitem__(self, index):
        im, gt, default_height, default_width, data_id = self.pull_item(index)
        if self.training is True:
            return im, gt, default_height, default_width, data_id
        else:
            return im, default_height, default_width, data_id
        

    def __len__(self):
        return len(self.ids)

    def get_ids(self):
        image_list = ld(self.image_root)
        if ".ipynb_checkpoints" in image_list:
            image_list.remove(".ipynb_checkpoints")
        ids = [filename.split(".")[0] for filename in image_list]
        return ids

    def crop_and_resize_image(self, img, img_after_crop_h, img_after_crop_w, i, j):
        c_img = img[
            img_after_crop_h * i: img_after_crop_h * (i + 1) + 100,
            img_after_crop_w * j: img_after_crop_w * (j + 1) + 100,
            :]
        cr_img = cv2.resize(c_img, dsize=(
            self.resize_size, self.resize_size), interpolation=cv2.INTER_NEAREST)
        return cr_img

    def crop_and_resize_annotations_refinedet(self, target_lines, img_after_crop_h, img_after_crop_w, default_height, default_width, i, j):
        targets = []
        for target_line in target_lines:
            target_line = target_line.split("\n")[0]
            target_line = target_line.split(" ")
            # target x1,x2,y1,y2
            t_x1 = float(target_line[0])
            t_x2 = float(target_line[2])
            t_y1 = float(target_line[1])
            t_y2 = float(target_line[3])
            # image x1,x2,y1,y2
            i_x1 = j / self.crop_num[1]
            i_x2 = (j + 1) / self.crop_num[1] + 100 / default_width
            i_y1 = i / self.crop_num[0]
            i_y2 = (i + 1) / self.crop_num[0] + 100 / default_height
            if (i_y1 < t_y1 and t_y2 < i_y2):  # for y
                if (i_x1 < t_x1 and t_x2 < i_x2):  # for x
                    if (i == self.crop_num[0] - 1 and j == self.crop_num[1] - 1):
                        targets.append(np.asarray([(t_x1 - i_x1) * default_width / (img_after_crop_w),
                                                    (t_y1 - i_y1) * default_height /
                                                    (img_after_crop_h),
                                                    (t_x2 - i_x1) * default_width /
                                                    (img_after_crop_w),
                                                    (t_y2 - i_y1) * default_height / (img_after_crop_h), float(target_line[4])], dtype="float32"))
                    elif (i == self.crop_num[0] - 1):
                        targets.append(np.asarray([(t_x1 - i_x1) * default_width / (img_after_crop_w + 100),
                                                    (t_y1 - i_y1) * default_height /
                                                    (img_after_crop_h),
                                                    (t_x2 - i_x1) * default_width /
                                                    (img_after_crop_w + 100),
                                                    (t_y2 - i_y1) * default_height / (img_after_crop_h), float(target_line[4])], dtype="float32"))
                    elif (j == self.crop_num[1] - 1):
                        targets.append(np.asarray([(t_x1 - i_x1) * default_width / (img_after_crop_w),
                                                    (t_y1 - i_y1) * default_height /
                                                    (img_after_crop_h + 100),
                                                    (t_x2 - i_x1) * default_width /
                                                    (img_after_crop_w),
                                                    (t_y2 - i_y1) * default_height / (img_after_crop_h + 100), float(target_line[4])], dtype="float32"))
                    else:
                        targets.append(np.asarray([(t_x1 - i_x1) * default_width / (img_after_crop_w + 100),
                                                    (t_y1 - i_y1) * default_height /
                                                    (img_after_crop_h + 100),
                                                    (t_x2 - i_x1) * default_width /
                                                    (img_after_crop_w + 100),
                                                    (t_y2 - i_y1) * default_height / (img_after_crop_h + 100), float(target_line[4])], dtype="float32"))
        return targets
    
    
    def crop_and_resize_annotations_faster_rcnn(self, target_lines, img_after_crop_h, img_after_crop_w, default_height, default_width, i, j):
        boxes = []
        labels = []
        for target_line in target_lines:
            target_line = target_line.split("\n")[0]
            target_line = target_line.split(" ")
            # target x1,x2,y1,y2
            t_x1 = float(target_line[0])
            t_x2 = float(target_line[2])
            t_y1 = float(target_line[1])
            t_y2 = float(target_line[3])
            # image x1,x2,y1,y2
            i_x1 = j / self.crop_num[1]
            i_x2 = (j + 1) / self.crop_num[1] + 100 / default_width
            i_y1 = i / self.crop_num[0]
            i_y2 = (i + 1) / self.crop_num[0] + 100 / default_height
            if (i_y1 < t_y1 and t_y2 < i_y2):  # for y
                if (i_x1 < t_x1 and t_x2 < i_x2):  # for x
                    if (i == self.crop_num[0] - 1 and j == self.crop_num[1] - 1):
                        boxes.append(np.asarray([(t_x1 - i_x1) * default_width * (self.resize_size / (img_after_crop_w)),
                                                 (t_y1 - i_y1) * default_height *
                                                 (self.resize_size /
                                                  (img_after_crop_h)),
                                                 (t_x2 - i_x1) * default_width *
                                                 (self.resize_size /
                                                  (img_after_crop_w)),
                                                 (t_y2 - i_y1) * default_height * (self.resize_size / (img_after_crop_h))], dtype="float32"))
                        labels.append(int(target_line[4]))
                    elif (i == self.crop_num[0] - 1):
                        boxes.append(np.asarray([(t_x1 - i_x1) * default_width * (self.resize_size / (img_after_crop_w + 100)),
                                                 (t_y1 - i_y1) * default_height *
                                                 (self.resize_size /
                                                  (img_after_crop_h)),
                                                 (t_x2 - i_x1) * default_width *
                                                 (self.resize_size /
                                                  (img_after_crop_w + 100)),
                                                 (t_y2 - i_y1) * default_height * (self.resize_size / (img_after_crop_h))], dtype="float32"))
                        labels.append(int(target_line[4]))
                    elif (j == self.crop_num[1] - 1):
                        boxes.append(np.asarray([(t_x1 - i_x1) * default_width * (self.resize_size / (img_after_crop_w)),
                                                 (t_y1 - i_y1) * default_height *
                                                 (self.resize_size /
                                                  (img_after_crop_h + 100)),
                                                 (t_x2 - i_x1) * default_width *
                                                 (self.resize_size /
                                                  (img_after_crop_w)),
                                                 (t_y2 - i_y1) * default_height * (self.resize_size / (img_after_crop_h + 100))], dtype="float32"))
                        labels.append(int(target_line[4]))
                    else:
                        boxes.append(np.asarray([(t_x1 - i_x1) * default_width * (self.resize_size / (img_after_crop_w + 100)),
                                                 (t_y1 - i_y1) * default_height *
                                                 (self.resize_size /
                                                  (img_after_crop_h + 100)),
                                                 (t_x2 - i_x1) * default_width *
                                                 (self.resize_size /
                                                  (img_after_crop_w + 100)),
                                                 (t_y2 - i_y1) * default_height * (self.resize_size / (img_after_crop_h + 100))], dtype="float32"))
                        labels.append(int(target_line[4]))
        return boxes, labels

        
    def pull_item(self, index):
        data_id = self.ids[index]

        if self.training is True:
            with open(pj(self.target_root, data_id + ".txt")) as f:
                target_lines = f.readlines()

        img = np.asarray(Image.open(pj(self.image_root, data_id + ".png")))
        default_height, default_width, default_channels = img.shape
        img = img.astype("float32")
        img = cv2.normalize(img, img, alpha=0, beta=1,
                            norm_type=cv2.NORM_MINMAX)

        cropped_img = []
        cropped_target = []
        cropped_data_id = []
        img_after_crop_h = int(default_height / self.crop_num[0])
        img_after_crop_w = int(default_width / self.crop_num[1])
        for i in range(self.crop_num[0]):
            for j in range(self.crop_num[1]):
                cr_img = self.crop_and_resize_image(
                    img, img_after_crop_h, img_after_crop_w, i, j)
                if self.training is True:
                    if self.model_name == "RefineDet":
                        targets = self.crop_and_resize_annotations_refinedet(
                            target_lines, img_after_crop_h, img_after_crop_w, default_height, default_width, i, j)
                        cropped_target.append(torch.from_numpy(np.asarray(targets)))
                    elif self.model_name == "Faster_RCNN":
                        boxes, labels = self.crop_and_resize_annotations_faster_rcnn(
                        target_lines, img_after_crop_h, img_after_crop_w, default_height, default_width, i, j)

                        targets = {}
                        targets.update(
                            {"boxes": torch.from_numpy(np.asarray(boxes))})
                        targets.update(
                            {"labels": torch.from_numpy(np.asarray(labels))})
                        cropped_target.append(targets)

                cropped_img.append(cr_img.transpose(2, 0, 1))
                cropped_data_id.append([self.ids[index], (i, j)])
        
        if self.model_name == "Faster_RCNN":
            cropped_target = np.asarray(cropped_target)
            cropped_data_id = np.asarray(cropped_data_id)

        return np.asarray(cropped_img), cropped_target, default_height, default_width, cropped_data_id


def collate_fn(batch):
    return tuple(zip(*batch))