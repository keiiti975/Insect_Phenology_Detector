import cv2
import copy
import numpy as np
from os import listdir as ld
from os.path import join as pj
from PIL import Image
import warnings
import torch
import torch.utils.data as data
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

class insects_dataset_from_voc_style_txt(data.Dataset):

    def __init__(self, image_root, resize_size, crop_num, training=False, target_root=None, method_crop="SPREAD_ALL_OVER", method_aug=None, model_detect_type="all"):
        """
            initializer
            Args:
                - image_root: str, image folder
                - resize_size: int, resized size of image
                - crop_num: (int, int), (height_crop_num, width_crop_num)
                - training: bool
                - target_root: str, if training is True, this is must
                - method_crop: str, choice ["SPREAD_ALL_OVER", "RANDOM"]
                - method_aug: [str, ...], adopt augmentation list
                - model_detect_type: str, choice ["all", "each", "det2cls"]
        """
        self.image_root = image_root
        self.resize_size = resize_size
        self.crop_num = crop_num
        self.training = training
        self.method_crop = method_crop
        self.method_aug = method_aug
        if training is True:
            if target_root is None:
                warnings.warn("Error! if training is True, target_root is must.")
            else:
                self.target_root = target_root
        if model_detect_type=="all":
            self.lbl_to_name = {
                0: "Insect"
            }
            self.name_to_lbl = {
                "Insect": 0
            }
        elif model_detect_type=="each":
            print("not implemented! insect_dataset_from_voc_style_txt.__init__")
        elif model_detect_type=="det2cls":
            self.lbl_to_name = {
                0: "Insect",
                1: "Other"
            }
            self.name_to_lbl = {
                "Insect": 0,
                "Other": 1
            }
        else:
            warnings.warn("Error! choice from [all, each, det2cls].")
        self.ids = self.get_ids(image_root)

    def __getitem__(self, index):
        """
            get item with index
            Args:
                - index: int, index of ids
        """
        # load image
        image, default_height, default_width = self.load_image(index)
        
        if self.training is True:
            # load annotation
            bbs_list = self.load_bbs_list(index, default_height, default_width)
            bbs = BoundingBoxesOnImage(bbs_list, shape=image.shape)
            
            # crop and resize image, annotation
            if self.method_crop == "SPREAD_ALL_OVER":
                image_crop, bbs_crop_list = self.crop_spread_all_over(image, bbs)
                bbs_crop = [BoundingBoxesOnImage(bbs_crop_list[idx], shape=image_crop[idx].shape) for idx in range(len(bbs_crop_list))]
            elif self.method_crop == "RANDOM":
                print("not implemented!: insect_dataset_from_voc_style_txt.__getitem__")
            
            # augment image, annotation
            if self.method_aug is not None:
                image_crop_aug, bbs_crop_aug_list = self.adopt_augmentation(image_crop, bbs_crop)
                bbs_crop_aug = [BoundingBoxesOnImage(bbs_crop_aug_list[idx], shape=image_crop_aug[idx].shape) for idx in range(len(bbs_crop_aug_list))]
            else:
                image_crop_aug = image_crop
                bbs_crop_aug = bbs_crop
            
            # create pytorch image, annotation
            image_crop_aug = image_crop_aug.transpose(0, 3, 1, 2)
            target = self.create_pytorch_annotation(bbs_crop_aug)
            
            return image_crop_aug, target, default_height, default_width, self.ids[index]
        else:
            # crop and resize image
            image_crop, _ = self.crop_spread_all_over(image)
            # set id for cropped_image
            data_ids = []
            for i in range(self.crop_num[0]):
                for j in range(self.crop_num[1]):
                    data_ids.append([self.ids[index], (i, j)])
                    
            return image_crop, default_height, default_width, data_ids
        
        
    def __len__(self):
        return len(self.ids)

    def get_ids(self, image_root):
        """
            load image id
            Args:
                - image_root: str, image folder
        """
        image_list = ld(image_root)
        if ".ipynb_checkpoints" in image_list:
            image_list.remove(".ipynb_checkpoints")
        ids = [filename.split(".")[0] for filename in image_list]
        return ids
    
    def load_image(self, index):
        """
            load image with index
            Args:
                - index: int, index of ids
        """
        data_id = self.ids[index]
        
        # load img and normalize
        image = np.asarray(Image.open(pj(self.image_root, data_id + ".png")))
        image = image.astype("float32")
        image = cv2.normalize(image, image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # get default height, width
        default_height, default_width, _ = image.shape
        
        return image, default_height, default_width
    
    def load_bbs_list(self, index, default_height, default_width):
        """
            load bounding box with index
            Args:
                - index: int, index of ids
                - default_height: int
                - default_width: int
        """
        data_id = self.ids[index]
        
        # load bounding box from file
        with open(pj(self.target_root, data_id + ".txt")) as f:
            target_lines = f.readlines()
        
        # create bbs_list
        bbs_list = []
        for target_line in target_lines:
            target_line = target_line.split("\n")[0]
            target_list = target_line.split(" ")
            x1 = float(target_list[0]) * default_width
            x2 = float(target_list[2]) * default_width
            y1 = float(target_list[1]) * default_height
            y2 = float(target_list[3]) * default_height
            bbs_list.append(BoundingBox(x1 = x1, x2 = x2, y1 = y1, y2 = y2, label = self.lbl_to_name[int(target_list[4])]))
        
        return bbs_list
    
    def crop_spread_all_over(self, image, bbs=None):
        """
            crop img, type == "SPREAD_ALL_OVER"
            Args:
                - img: np.array, shape == [1, height, width, channel]
                - bbs: imgaug BoundingBox
        """
        height_after_crop = int(image.shape[0] / self.crop_num[0])
        width_after_crop = int(image.shape[1] / self.crop_num[1])

        height_mov_ratio_per_crop = 1.0 / (self.crop_num[0] - 1)
        width_mov_ratio_per_crop = 1.0 / (self.crop_num[1] - 1)

        image_crop_list = []
        bbs_crop_list = []
        # create crop img
        for i in range(self.crop_num[0]):
            for j in range(self.crop_num[1]):
                # set augmentations
                aug_seq = iaa.Sequential([
                    iaa.CropToFixedSize(width=width_after_crop, height=height_after_crop, position=(1.0 - width_mov_ratio_per_crop * j, 1.0 - height_mov_ratio_per_crop * i)),
                    iaa.Resize({"width": self.resize_size, "height": self.resize_size})
                ])
                # augment img and target
                if bbs is not None:
                    image_crop, bbs_crop = aug_seq(image=image, bounding_boxes=bbs)
                    # check coord in img shape
                    bbs_crop_before_check = bbs_crop.bounding_boxes
                    bbs_crop_after_check = copy.copy(bbs_crop.bounding_boxes)
                    for bb in bbs_crop_before_check:
                        if bb.is_fully_within_image(image_crop.shape):
                            pass
                        else:
                            bbs_crop_after_check.remove(bb)
                    # append img and target
                    if len(bbs_crop_after_check) > 0:
                        image_crop_list.append(image_crop)
                        bbs_crop_list.append(bbs_crop_after_check)
                else:
                    image_crop = aug_seq(image=image)

        return np.array(image_crop_list), bbs_crop_list
    
    def crop_random(image, bbs=None):
        """
            crop img, type == "RANDOM"
            Args:
                - img: np.array, shape == [1, height, width, channel]
                - bbs: imgaug BoundingBox
        """
        height_after_crop = int(image.shape[0] / self.crop_num[0])
        width_after_crop = int(image.shape[1] / self.crop_num[1])

        image_aug_list = []
        bbs_aug_list = []
        for i in range(self.crop_num[0] * self.crop_num[1]):
            # set augmentations
            aug_seq = iaa.Sequential([
                iaa.CropToFixedSize(width=width_after_crop, height=height_after_crop, position="uniform"),
                iaa.Resize({"width": self.resize_size, "height": self.resize_size})
            ])
            # augment img and target
            if bbs is not None:
                image_aug, bbs_aug = aug_seq(image=image, bounding_boxes=bbs)
                # check coord in img shape
                bbs_aug_before_check = bbs_aug.bounding_boxes
                bbs_aug_after_check = copy.copy(bbs_aug.bounding_boxes)
                for bb in bbs_aug_before_check:
                    if bb.is_fully_within_image(image_aug.shape):
                        pass
                    else:
                        bbs_aug_after_check.remove(bb)
                # append img and target
                if len(bbs_aug_after_check) > 0:
                    image_aug_list.append(image_aug)
                    bbs_aug_list.append(bbs_aug_after_check)
            else:
                image_aug = aug_seq(image=image)

        return np.array(image_aug_list), bbs_aug_list
    
    def adopt_augmentation(self, image_crop, bbs_crop):
        """
            adopt augmentation to image_crop, bbs_crop
            Args:
                - image_crop: np.array, shape == [crop_num, height, width, channels], cropped images
                - bbs_crop: [BoundingBoxesOnImage, ...], imgaug bounding box
                - method_aug: [str, ...], adopt augmentation list
        """
        aug_list = []
        # create augmentation
        for augmentation in self.method_aug:
            if augmentation == "HorizontalFlip":
                aug_list.append(iaa.Fliplr(0.5))
            elif augmentation == "VerticalFlip":
                aug_list.append(iaa.Flipud(0.5))
            elif augmentation == "Rotate":
                aug_list.append(iaa.Rotate((-45, 45)))
            else:
                print("not implemented!: insect_dataset_from_voc_style_txt.adopt_augmentation")

        aug_seq = iaa.SomeOf(1, aug_list)

        image_crop_aug = []
        bbs_crop_aug = []
        # adopt augmentation
        for im_crop, bb_crop in zip(image_crop, bbs_crop):
            im_crop_aug, bb_crop_aug = aug_seq(image=im_crop, bounding_boxes=bb_crop)
            # check coord in im_crop_aug shape
            bb_crop_aug_before_check = bb_crop_aug.bounding_boxes
            bb_crop_aug_after_check = copy.copy(bb_crop_aug.bounding_boxes)
            for bb in bb_crop_aug_before_check:
                if bb.is_fully_within_image(im_crop_aug.shape):
                    pass
                else:
                    bb_crop_aug_after_check.remove(bb)
            # append im_crop_aug and bb_crop_aug
            if len(bb_crop_aug_after_check) > 0:
                image_crop_aug.append(im_crop_aug)
                bbs_crop_aug.append(bb_crop_aug_after_check)

        return np.array(image_crop_aug), bbs_crop_aug
    
    def create_pytorch_annotation(self, bbs_crop_aug):
        """
            extract bounding box and convert to pytorch style
            Args:
                - bbs_crop_aug: [BoundingBoxesOnImage, ...], imgaug bounding box
        """
        target = []
        for bb_crop_aug in bbs_crop_aug:
            target_per_image = []
            for elem_bb in bb_crop_aug:
                x1 = elem_bb.x1 / float(self.resize_size)
                y1 = elem_bb.y1 / float(self.resize_size)
                x2 = elem_bb.x2 / float(self.resize_size)
                y2 = elem_bb.y2 / float(self.resize_size)
                lbl = int(self.name_to_lbl[elem_bb.label])
                target_per_image.append([x1, y1, x2, y2, lbl])
            target.append(torch.Tensor(target_per_image))
        return target


def collate_fn(batch):
    return tuple(zip(*batch))