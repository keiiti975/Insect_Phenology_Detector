from os import listdir as ld
from os.path import join as pj
import numpy as np
from PIL import Image
from evaluation.Object_Detection_Metrics.lib.BoundingBox import BoundingBox
from evaluation.Object_Detection_Metrics.lib.BoundingBoxes import BoundingBoxes
from evaluation.Object_Detection_Metrics.lib.Evaluator import Evaluator
from evaluation.Object_Detection_Metrics.lib.utils import *


class Voc_Evaluater:

    def __init__(self, image_root, target_root):
        self.image_root = image_root
        self.target_root = target_root
        self.default_img_size_dic = self.get_default_img_size_dic()
        self.gtBoundingBoxes = BoundingBoxes()
        self.myBoundingBoxes = BoundingBoxes()
        self.initialize_ground_truth()
        
    def get_default_img_size_dic(self):
        default_img_size_dic = {}
        anno_file_names = ld(self.target_root)
        anno_file_names = [filename.split('.')[0] for filename in anno_file_names if filename != ".ipynb_checkpoints"]
        for anno_file_name in anno_file_names:
            img = np.asarray(Image.open(pj(self.image_root, anno_file_name + ".png")))
            img_height, img_width, _ = img.shape
            default_img_size_dic.update({anno_file_name: [img_height, img_width]})
        return default_img_size_dic

    def initialize_ground_truth(self):
        gtBoundingBoxes = BoundingBoxes()
        print("initialize evaluater ...")
        anno_file_names = ld(self.target_root)
        anno_file_names = [filename.split('.')[0] for filename in anno_file_names if filename != ".ipynb_checkpoints"]
        for anno_file_name in anno_file_names:
            img_height, img_width = self.default_img_size_dic[anno_file_name]
            with open(pj(self.target_root, anno_file_name + ".txt"), mode='r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split("\n")[0]
                    line = line.split(" ")
                    x_min = float(line[0]) * img_width
                    y_min = float(line[1]) * img_height
                    x_max = float(line[2]) * img_width
                    y_max = float(line[3]) * img_height
                    label = str(line[4])
                    bb = BoundingBox(imageName = anno_file_name, classId = label, 
                                     x = x_min, y = y_min, w = x_max, h = y_max, typeCoordinates=CoordinatesType.Absolute, 
                                     bbType=BBType.GroundTruth, format=BBFormat.XYX2Y2)
                    """
                    print("-----")
                    print("anno_file_name = " + anno_file_name)
                    print("label = " + label)
                    print("x_min = " + str(x_min))
                    print("y_min = " + str(y_min))
                    print("x_max = " + str(x_max))
                    print("y_max = " + str(y_max))
                    print("typeCoordinates = " + str(CoordinatesType.Absolute))
                    print("bbType = " + str(BBType.GroundTruth))
                    print("format = " + str(BBFormat.XYX2Y2))
                    print("imgSize = x: " + str(img_width) + " y: " + str(img_height))
                    print("#####")
                    """
                    gtBoundingBoxes.addBoundingBox(bb)
        self.gtBoundingBoxes = gtBoundingBoxes
    
    def set_result(self, result):
        myBoundingBoxes = self.gtBoundingBoxes
        print("setting result ...")
        for data_id, cls_dets_per_class in result.items():
            img_height, img_width = self.default_img_size_dic[data_id]
            for cls_label, detections in cls_dets_per_class.items():
                label = str(cls_label)
                for detection in detections:
                    x_min = float(detection[0])
                    y_min = float(detection[1])
                    x_max = float(detection[2])
                    y_max = float(detection[3])
                    conf = float(detection[4])
                    bb = BoundingBox(imageName = data_id, classId = label, 
                                     x = x_min, y = y_min, w = x_max, h = y_max, typeCoordinates=CoordinatesType.Absolute, 
                                     bbType=BBType.Detected, format=BBFormat.XYX2Y2, classConfidence=conf)
                    """
                    print("-----")
                    print("anno_file_name = " + data_id)
                    print("label = " + label)
                    print("x_min = " + str(x_min))
                    print("y_min = " + str(y_min))
                    print("x_max = " + str(x_max))
                    print("y_max = " + str(y_max))
                    print("typeCoordinates = " + str(CoordinatesType.Absolute))
                    print("bbType = " + str(BBType.Detected))
                    print("format = " + str(BBFormat.XYX2Y2))
                    print("classConfidence = " + str(conf))
                    print("imgSize = x: " + str(img_width) + " y: " + str(img_height))
                    print("#####")
                    """
                    myBoundingBoxes.addBoundingBox(bb)
        self.myBoundingBoxes = myBoundingBoxes
    
    def get_eval_metrics(self):
        evaluator = Evaluator()
        # Get metrics with PASCAL VOC metrics
        metricsPerClass = evaluator.GetPascalVOCMetrics(
            self.myBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
            IOUThreshold=0.3,  # IOU threshold
            method=MethodAveragePrecision.EveryPointInterpolation)  # As the official matlab code
        return metricsPerClass
