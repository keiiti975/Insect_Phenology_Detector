import numpy as np
from tqdm import tqdm
import torch


def test_prediction(model, data_loader, crop_num, num_classes, nms_thresh=0.3):
    """
        get refinedet prediction with classification (dataset = insects_dataset_from_voc_style_txt)
        - model: pytorch model
        - data_loader: insects_dataset_from_voc_style_txt
        - crop_num: (int, int)
        - num_classes: (int), front_class + background_class
        - nms_thresh: (float)
    """
    # set refinedet to eval mode
    model.eval()
    result = {}

    # image_id is needed because order of all_boxes is different from order of recs
    image_id = []
    for images, default_height, default_width, data_ids in tqdm(data_loader):
        images = images[0]
        default_height = default_height[0]
        default_width = default_width[0]
        data_ids = data_ids[0]
        image_id.append(data_ids[0][0])
        print("detecting ... : " + str(data_ids[0][0]))

        # define cropped image height,width
        img_after_crop_h = int(default_height / crop_num[0])
        img_after_crop_w = int(default_width / crop_num[1])

        # class detection for all image
        cls_dets_per_class = [[] for i in range(num_classes-1)]

        # estimate for cropped image
        for i in range(images.shape[0]):
            image = images[i:i + 1]
            image = torch.from_numpy(image).cuda()

            # estimate
            detections = model(image)
            for j in range(num_classes-1):
                detection = detections[0][j+1].detach().cpu().numpy()

                boxes = detection[:, 1:]
                scores = detection[:, 0]

                if data_ids[i][1][0] == crop_num[0] - 1 and data_ids[i][1][1] == crop_num[1] - 1:
                    boxes[:, 0] = (boxes[:, 0] * img_after_crop_w) + \
                        data_ids[i][1][1] / crop_num[1] * default_width
                    boxes[:, 1] = (boxes[:, 1] * img_after_crop_h) + \
                        data_ids[i][1][0] / crop_num[0] * default_height
                    boxes[:, 2] = (boxes[:, 2] * img_after_crop_w) + \
                        data_ids[i][1][1] / crop_num[1] * default_width
                    boxes[:, 3] = (boxes[:, 3] * img_after_crop_h) + \
                        data_ids[i][1][0] / crop_num[0] * default_height
                elif data_ids[i][1][0] == crop_num[0] - 1:
                    boxes[:, 0] = (boxes[:, 0] * (img_after_crop_w + 100)) + \
                        data_ids[i][1][1] / crop_num[1] * default_width
                    boxes[:, 1] = (boxes[:, 1] * img_after_crop_h) + \
                        data_ids[i][1][0] / crop_num[0] * default_height
                    boxes[:, 2] = (boxes[:, 2] * (img_after_crop_w + 100)) + \
                        data_ids[i][1][1] / crop_num[1] * default_width
                    boxes[:, 3] = (boxes[:, 3] * img_after_crop_h) + \
                        data_ids[i][1][0] / crop_num[0] * default_height
                elif data_ids[i][1][1] == crop_num[1] - 1:
                    boxes[:, 0] = (boxes[:, 0] * img_after_crop_w) + \
                        data_ids[i][1][1] / crop_num[1] * default_width
                    boxes[:, 1] = (boxes[:, 1] * (img_after_crop_h + 100)) + \
                        data_ids[i][1][0] / crop_num[0] * default_height
                    boxes[:, 2] = (boxes[:, 2] * img_after_crop_w) + \
                        data_ids[i][1][1] / crop_num[1] * default_width
                    boxes[:, 3] = (boxes[:, 3] * (img_after_crop_h + 100)) + \
                        data_ids[i][1][0] / crop_num[0] * default_height
                else:
                    boxes[:, 0] = (boxes[:, 0] * (img_after_crop_w + 100)) + \
                        data_ids[i][1][1] / crop_num[1] * default_width
                    boxes[:, 1] = (boxes[:, 1] * (img_after_crop_h + 100)) + \
                        data_ids[i][1][0] / crop_num[0] * default_height
                    boxes[:, 2] = (boxes[:, 2] * (img_after_crop_w + 100)) + \
                        data_ids[i][1][1] / crop_num[1] * default_width
                    boxes[:, 3] = (boxes[:, 3] * (img_after_crop_h + 100)) + \
                        data_ids[i][1][0] / crop_num[0] * default_height
                # class detection for cropped image
                cropped_cls_dets = np.hstack(
                    (boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                # collect detection result
                cls_dets_per_class[j].extend(cropped_cls_dets)

        dic_cls_dets_per_class = {}
        for i in range(num_classes-1):
            cls_dets_per_class[i] = np.asarray(cls_dets_per_class[i])
            keep = nms(cls_dets_per_class[i], nms_thresh)
            cls_dets_per_class[i] = cls_dets_per_class[i][keep]
            dic_cls_dets_per_class.update({i: cls_dets_per_class[i]})
        result.update({data_ids[0][0]: dic_cls_dets_per_class})

    return result


# Malisiewicz et al.
def nms(boxes, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return pick
