import numpy as np
import torch
from model.resnet.predict import test_classification


def evaluate(result, ground_truth, name2lbl, ovthresh=0.3):
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
    pred_lbls, BB, confidence, image_ids = create_global_pred(result)
    gt_dict, npos = create_imwise_gt(ground_truth)

    tp, fp = compute_tp_fp(pred_lbls, BB, confidence,
                           image_ids, gt_dict, npos, name2lbl, ovthresh)

    recall, precision, avg_precision = summarize_precision_recall(fp, tp, npos)
    return recall, precision, avg_precision, gt_dict


def create_global_pred(detections):
    pred_lbls, BB, confidence, image_ids, aps = [], [], [], [], []

    for im_id, detection in detections.items():
        for box in detection['coord']:
            image_ids.append(im_id)
            confidence.append(box[4])
            BB.append(box[:4])
        for lbl in detection['output_lbl']:
            pred_lbls.append(lbl)

    confidence = np.asarray(confidence, dtype="float32")
    BB = np.asarray(BB, dtype="int32")
    pred_lbls = np.asarray(pred_lbls, dtype="int32")

    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    pred_lbls = pred_lbls[sorted_ind]
    image_ids = [image_ids[x] for x in sorted_ind]
    return pred_lbls, BB, confidence, image_ids


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


def compute_tp_fp(pred_lbls, BB, confidence, image_ids, gt_dict, npos, name2lbl, ovthresh=0.3):
    n_pred = len(image_ids)
    true_positive = np.zeros(n_pred)
    false_positive = np.zeros(n_pred)

    for pred_id in range(n_pred):
        ground_truth = gt_dict[image_ids[pred_id]]
        pred_bbox = BB[pred_id, :].astype(float)
        gt_lbls = np.asarray([name2lbl[gt_name]
                              for gt_name in ground_truth['default_name']])
        ovmax, jmax = iou(pred_bbox, ground_truth['bbox'].astype(float))
        update_tfp_results(ground_truth, gt_lbls, pred_lbls,
                           true_positive, false_positive, pred_id, ovmax, jmax, 0.5)

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


def update_tfp_results(ground_truth, gt_lbls, pred_lbls, tp, fp, pred_id, ovmax, jmax, ovthresh):
    if ovmax > ovthresh:
        if not ground_truth['difficult'][jmax] and gt_lbls[jmax] >= 0:
            if not ground_truth['det'][jmax] and (gt_lbls[jmax] == pred_lbls[pred_id]):
                tp[pred_id] = 1.
                ground_truth['det'][jmax] = 1
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


def get_cls_gt(out_box, gt_box, gt_lbl, name2lbl, thr=.3):
    res = []
    for bb in out_box:
        x = iou(bb, gt_box)
        if x[0] > thr:
            gtidx = x[1]
            res.append(gt_lbl[gtidx])
        else:
            res.append("spider")
    return torch.LongTensor([name2lbl[i] for i in res]).cuda()


def get_cls_accuracy_per_class(insect_dataset, result, gt_dict, name2lbl, add_divide_model=False, n_class_when_not_use_divide_model=7):
    accs = []
    all_out = []
    correct_gt = []
    correct_lbl = []
    for image_id, imgs in insect_dataset.items():
        out = result[image_id]["output_lbl"]
        out_box = result[image_id]["coord"]
        gt_box = gt_dict[image_id]["bbox"]
        gt_lbl = gt_dict[image_id]['default_name']
        gt_lbl2 = np.asarray([name2lbl[lbl] for lbl in gt_lbl])
        correct_gt.extend(gt_lbl2[gt_lbl2 != -1])
        gt_lbl = get_cls_gt(out_box, gt_box, gt_lbl, name2lbl)
        if add_divide_model is True:
            gt_lbl = gt_lbl.cpu().numpy()
            out = np.asarray([output for i, output in enumerate(out) if result[image_id]["divide_lbl"][i] == 0])
            gt_lbl = np.asarray([label for i, label in enumerate(gt_lbl) if result[image_id]["divide_lbl"][i] == 0])
            acc = (out == gt_lbl).mean()
            accs.append(acc)
            all_out.extend(out)
            correct_lbl.extend(gt_lbl[(out == gt_lbl)])
        else:
            gt_lbl = gt_lbl.cpu().numpy()
            out_mask = out == (n_class_when_not_use_divide_model - 1)
            out = np.asarray([output for i, output in enumerate(out) if out_mask[i] == False])
            gt_lbl = np.asarray([label for i, label in enumerate(gt_lbl) if out_mask[i] == False])
            acc = (out == gt_lbl).mean()
            accs.append(acc)
            all_out.extend(out)
            correct_lbl.extend(gt_lbl[(out == gt_lbl)])

    _, out_count = np.unique(all_out, return_counts=True)
    _, gt_count = np.unique(correct_gt, return_counts=True)
    _, lbl_count = np.unique(correct_lbl, return_counts=True)
    if add_divide_model is True:
        recalls = lbl_count/gt_count
    else:
        recalls = lbl_count/gt_count[:-1]
    precisions = lbl_count/out_count
    return accs, recalls, precisions
