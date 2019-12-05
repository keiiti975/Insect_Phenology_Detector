import numpy as np


def evaluate(detections, ground_truth, ovthresh=0.3):
    """
        Input:
            detection:
            ground_truth:
            ovthresh:
        Output:
            recall:
            precision:
            avg_precision:
            gt_dict:
    """
    BB, confidence, image_ids = create_global_pred(detections)
    gt_dict, npos = create_imwise_gt(ground_truth)

    tp, fp = compute_tp_fp(image_ids, confidence, BB, gt_dict, npos, ovthresh)

    recall, precision, avg_precision = summarize_precision_recall(fp, tp, npos)
    return recall, precision, avg_precision, gt_dict


def create_global_pred(detections):
    """
        Creates the global ordering of detection by confidence scores

        Inputs:
            detections:
        outputs:
            BB:
            confidence:
            image_ids:
    """
    aps, image_ids, confidence, BB = [], [], [], []

    for im_id, detection in detections.items():
        for box in detection:
            image_ids.append(im_id)
            confidence.append(box[4])
            BB.append(box[:4])

    confidence = np.asarray(confidence, dtype="float32")
    BB = np.asarray(BB, dtype="int32")

    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    return BB, confidence, image_ids


def create_imwise_gt(groundtruth_list):
    """
        Input:
            groundtruth_list:
        Output:
            groundtruth_dict:
            npos:
    """
    groundtruth_dict = {}
    npos = 0
    for im_id, gt_list in groundtruth_list.items():
        bbox = np.array([x['bbox'] for x in gt_list])
        difficult = np.array([x['difficult'] for x in gt_list]).astype(np.bool)
        default_name = np.asarray([x['default_name'] for x in gt_list])
        det = [False] * len(gt_list)
        npos = npos + sum(~difficult)

        groundtruth_dict[im_id] = {'bbox': bbox,
                                   'difficult': difficult,
                                   'det': det,
                                   'default_name': default_name
                                   }
    return groundtruth_dict, npos


def compute_tp_fp(image_ids, confidence, BB, gt_dict, npos, ovthresh=0.3):
    """
        Return true positive and false negatives.
        Update gt_dic[image_id]["det"] accordingly

        Input:
            image_ids:
            confidence:
            BB:
            gt_dict:
            npos:
            ovthresh:
        Output:
            true_positive:
            false_positive:
    """
    n_pred = len(image_ids)
    true_positive = np.zeros(n_pred)
    false_positive = np.zeros(n_pred)

    for pred_id in range(n_pred):
        img_gt = gt_dict[image_ids[pred_id]]
        pred_bbox = BB[pred_id, :].astype(float)
        gt_bboxes = img_gt['bbox'].astype(float)
        ovmax, jmax = iou(pred_bbox, gt_bboxes)
        update_tfp_results(img_gt, true_positive,
                           false_positive, pred_id, ovmax, jmax, ovthresh)

    return true_positive, false_positive


def iou(bb, BBGT):
    """
        Input:
            bb:
            BBGT:
        Output:
            ovmax:
            jmax:
    """
    if BBGT.size > 0:
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)
        return ovmax, jmax
    else:
        return -np.inf, None


def update_tfp_results(img_gt, tp, fp, pred_id, ovmax, jmax, ovthresh):
    """
        Input:
            img_gt:
            tp:
            fp:
            pred_id:
            ovmax:
            jmax:
            ovthresh:
    """
    if ovmax > ovthresh:
        if not img_gt['difficult'][jmax]:
            if not img_gt['det'][jmax]:
                tp[pred_id] = 1.
                img_gt['det'][jmax] = 1
            else:
                fp[pred_id] = 1.
    else:
        fp[pred_id] = 1.


def summarize_precision_recall(fp, tp, npos):
    """
        Input:
            fp:
            tp:
            npos:
        Output:
            prec:
            rec:
            ap:
    """
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    if npos == 0:
        rec = 0.
        prec = 0.
        ap = 0.
        return rec, prec, ap
    else:
        rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)
    return prec, rec, ap


def voc_ap(rec, prec):
    """
        Compute VOC AP given precision and recall.

        Input:
            rec:
            prec:
        Output:
            ap:
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_each_ap(gt_dict):
    for k, v in gt_dict.items():
        for dic_key, dic_value in v.items():
            if dic_key == "det":
                avg_precision = len(
                    [i for i in dic_value if i == 1]) / len(dic_value)
    return avg_precision
