import cv2
import numpy as np


def vis_detections(im, dets, class_name="insects", color_name="green", thresh=0.5):
    """Visual debugging of detections."""
    color = {"white":[255,255,255], "red":[255,0,0], "lime":[0,255,0], 
             "blue":[0,0,255], "yellow":[255,255,0], "fuchsia":[255,0,255],
            "aqua":[0,255,255], "gray":[128,128,128], "maroon":[128,0,0],
            "green":[0,128,0], "navy":[0,0,128], "olive":[128,128,0],
            "purple":[128,0,128], "teal":[0,128,128], "black":[0,0,0]}[color_name]
    for i in range(dets.shape[0]):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > 1.0:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], color, 2)
            cv2.putText(im, '%s: ground truth' % (class_name), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
        elif score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], color, 2)
            cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
    return im
