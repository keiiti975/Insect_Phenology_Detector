from os import listdir as ld
from os.path import join as pj
import numpy as np
from PIL import Image
from evaluation.Object_Detection_Metrics.lib.BoundingBox import BoundingBox
from evaluation.Object_Detection_Metrics.lib.BoundingBoxes import BoundingBoxes
from evaluation.Object_Detection_Metrics.lib.utils import *


class evaluater:

    def __init__(self, image_root, target_root):
        self.image_root = image_root
        self.target_root = target_root
        self.myBoundingBoxes = BoundingBoxes()
        self.initialize_ground_truth()

    def initialize_ground_truth(self):
        print("initialize evaluater ...")
        anno_file_names = ld(self.target_root)
        anno_file_names = [filename for filename in anno_file_names if filename != ".ipynb_checkpoints"]
        for anno_file_name in anno_file_names:
            img = np.asarray(Image.open(pj(self.image_root, anno_file_name + ".png")))
            img_height, img_width, _ = img.shape
            with open(pj(self.target_root, anno_file_name + ".txt"), mode='r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split("\n")[0]
                    line = line.split(" ")
                    x_min = float(line[0])
                    x_max = float(line[2])
                    y_min = float(line[1])
                    y_max = float(line[3])
                    label = str(line[4])
                    bb = BoundingBox(anno_file_name, label, x_min, y_min, x_max, y_max, \
                    typeCoordinates=CoordinatesType.Relative, bbType=BBType.GroundTruth, \
                    format=BBFormat.XYX2Y2, imgSize=(img_width, img_height))
                    self.myBoundingBoxes.addBoundingBox(bb)
                    
    def set_result(self, result):
        print("setting result ...")
        for data_id, cls_dets_per_class in result.items():
            for cls_label, detections in cls_dets_per_class.items():
                x_min = 
