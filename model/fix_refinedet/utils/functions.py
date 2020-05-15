import torch


def point_form(boxes):
    """Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center_form boxes.
    Return:
        boxes: (tensor) Converted (xmin, ymin, xmax, ymax) form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)  # xmax, ymax


def center_form(boxes):
    """Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center form ground truth data.
    Args:
        boxes: (tensor) point_form boxes.
    Return:
        boxes: (tensor) Converted (cx, cy, w, h) form of boxes.
    """
    return torch.cat([(boxes[:, 2:] + boxes[:, :2]) / 2,  # cx, cy
                      boxes[:, 2:] - boxes[:, :2]], 1)  # w, h


def intersect(box_a, box_b):
    """We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
        box_a: (tensor) bounding boxes, Shape: [A,4].
        box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
        (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects, 4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors, 4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def refine_match(overlap_thresh, target_boxes, prior_data, variances, target_labels, loc_t, conf_t, idx, arm_loc_data=None):
    """Match each arm bbox with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        overlap_thresh: (float) The overlap threshold used when mathing boxes.
        target_boxes: (tensor) Ground truth boxes, Shape: [num_objects, 4].
        prior_data: (tensor) Prior boxes from priorbox layers, Shape: [num_priors, 4].
        variances: (tensor) Variances for [dist(xmax-xmin, ymax-ymin), ymax].
        target_labels: (tensor) All the class labels for the image, Shape: [num_objects].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
        arm_loc_data.shape == [num_priors, 4]
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    if arm_loc_data is None:
        overlaps = jaccard(target_boxes, point_form(prior_data))
    else:
        prior_data_decoded_by_arm = decode_location_data(arm_loc_data, prior_data=prior_data, variances=variances)
        prior_data_decoded_by_arm = point_form(prior_data_decoded_by_arm)
        overlaps = jaccard(target_boxes, prior_data_decoded_by_arm)
    # (Bipartite Matching)
    # best_prior_overlap_for_ground_truth.shape == [num_objects, 1]
    # best_prior_idx_for_ground_truth.shape == [num_objects, 1]
    # best prior for each ground truth
    best_prior_overlap_for_ground_truth, best_prior_idx_for_ground_truth = overlaps.max(1, keepdim=True)
    # best_ground_truth_overlap_for_prior.shape == [1, num_priors]
    # best_ground_truth_idx_for_prior.shape == [1, num_priors]
    # best ground truth for each prior
    best_ground_truth_overlap_for_prior, best_ground_truth_idx_for_prior = overlaps.max(0, keepdim=True)
    best_ground_truth_idx_for_prior.squeeze_(0)
    best_ground_truth_overlap_for_prior.squeeze_(0)
    best_prior_idx_for_ground_truth.squeeze_(1)
    best_prior_overlap_for_ground_truth.squeeze_(1)
    best_ground_truth_overlap_for_prior.index_fill_(0, best_prior_idx_for_ground_truth, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j, best_prior_idx in enumerate(best_prior_idx_for_ground_truth):
        best_ground_truth_idx_for_prior[best_prior_idx] = j
    sorted_target_boxes_with_prior = target_boxes[best_ground_truth_idx_for_prior]  # Shape: [num_priors, 4]
    if arm_loc_data is None:
        sorted_target_label_with_prior = target_labels[best_ground_truth_idx_for_prior]  # Shape: [num_priors]
        loc = encode(center_form(sorted_target_boxes_with_prior), prior_data, variances)
    else:
        sorted_target_label_with_prior = target_labels[best_ground_truth_idx_for_prior] + 1  # Shape: [num_priors], 1 <= value <= num_classes
        loc = encode(center_form(sorted_target_boxes_with_prior), center_form(prior_data_decoded_by_arm), variances)
    sorted_target_label_with_prior[best_ground_truth_overlap_for_prior < overlap_thresh] = 0  # label as background
    loc_t[idx] = loc  # [num_priors, 4] encoded offsets to learn
    conf_t[idx] = sorted_target_label_with_prior  # [num_priors] top class label for each prior


def encode(target_boxes, prior_data, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in center form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4], center form
    """
    # dist b/t match center and prior's center
    g_cxcy = target_boxes[:, :2] - prior_data[:, :2]
    # encode variance
    g_cxcy = g_cxcy / (variances[0] * prior_data[:, 2:])
    # match wh / prior wh
    g_wh = target_boxes[:, 2:] / prior_data[:, 2:]
    # encode variance
    g_wh = torch.log(g_wh + 1e-5) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


def decode_location_data(loc_data, prior_data, variance=[0.1, 0.2]):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    function to correct prior_data with loc_data.
    Args:
        loc_data (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        prior_data (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions, center_form, (cx, cy, w, h)
            Shape: [num_priors,4]
    """
    boxes = torch.cat((
        prior_data[:, :2] + loc_data[:, :2] * variances[0] * prior_data[:, 2:],
        prior_data[:, 2:] * torch.exp(loc_data[:, 2:] * variances[1])), 1)
    return boxes


def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    keep = torch.zeros(scores.size(0), dtype=torch.int64)
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas
        union = (rem_areas - inter) + area[i]
        IoU = inter.float()/union.float()  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


def hard_negative_mining(loss_c, object_filter, negpos_ratio):
    """
        get filter of negative(==low loss) idx.
        number of negative idx <= min(3 * number of positive idx, num_priors - 1)
        Args:
            - loss_c: (tensor) classification loss
            - object_filter: (tensor) filter, object_cls == 1, background_cls == 0
            - negpos_ratio: (int) value that restricts number of negative idx
    """
    batch_size = object_filter.size(0)
    num_priors = object_filter.size(1)
    loss_c[object_filter.view(-1, 1)] = 0  # filter out pos boxes for now
    loss_c = loss_c.view(batch_size, -1)  # loss_c.shape == [batch_size, num_priors]
    idx_rank = torch.argsort(loss_c, dim=1, descending=True)  # idx_rank.shape == [batch_size, num_priors]
    num_pos = object_filter.long().sum(1, keepdim=True)
    num_neg = torch.clamp(self.negpos_ratio * num_pos, max=num_priors - 1)
    neg = idx_rank < num_neg.expand_as(idx_rank)  # neg.shape == [batch_size, num_priors]

    sum_pos = num_pos.detach().sum().float()
    return neg, sum_pos
