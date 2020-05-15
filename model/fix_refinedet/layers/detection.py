import torch
import torch.nn as nn
import torch.nn.functional as F
from model.refinedet.utils.functions import decode_location_data, nms

class Detect(nn.Module):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.background_label = 0
        self.top_k = 1000
        self.keep_top_k = 500

        # parameters for nms
        self.nms_thresh = 0.45
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = 0.01
        self.objectness_thresh = 0.01
        self.variance = [0.1, 0.2]

    def forward(self, arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data, prior_data):
        # get confidence from source conf_data
        arm_conf_data = F.softmax(arm_conf_data, dim=2)
        odm_conf_data = F.softmax(odm_conf_data, dim=2)

        arm_object_conf = arm_conf_data.data[:, :, 1]  # [:, :, 0] == non-object conf, [:, :, 1] == object conf
        no_object_filter = arm_object_conf <= self.objectness_thresh
        odm_conf_data[no_object_filter.expand_as(odm_conf_data)] = 0

        batch_size = odm_loc_data.size(0)
        num_priors = prior_data.size(0)
        result = torch.zeros(batch_size, self.num_classes, self.top_k, 5)
        # odm_conf_data.shape == [batch_size, num_priors, self.num_classes]
        # => odm_conf_preds.shape == [batch_size, self.num_classes, num_priors]
        odm_conf_preds = odm_conf_data.view(batch_size, num_priors, self.num_classes).transpose(2, 1)

        # decode predictions into bboxs
        for i in range(batch_size):
            prior_data_decoded_by_arm = decode_location_data(arm_loc_data[i], prior_data, self.variance)
            prior_data_decoded_by_odm = decode_location_data(odm_loc_data[i], prior_data_decoded_by_arm, self.variance)
            # for each class, perform nms
            # odm_conf_scores.shape == [self.num_classes, num_priors]
            # classes = background_class + other_classes
            odm_conf_scores = odm_conf_preds[i].clone()
            for cls_lbl in range(1, self.num_classes):
                conf_scores_filter_per_cls = odm_conf_scores[cls_lbl].gt(self.conf_thresh)
                filtered_conf_scores_per_cls = odm_conf_scores[cls_lbl][conf_scores_filter_per_cls]
                if filtered_conf_scores_per_cls.size(0) == 0:
                    continue
                prior_data_filter_per_cls = conf_scores_filter_per_cls.unsqueeze(1).expand_as(prior_data_decoded_by_odm)
                filtered_prior_data_per_cls = prior_data_decoded_by_odm[prior_data_filter_per_cls].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(filtered_prior_data_per_cls, filtered_conf_scores_per_cls, self.nms_thresh, self.top_k)
                result[i, cls_lbl, :count] = torch.cat((filtered_conf_scores_per_cls[ids[:count]].unsqueeze(1), filtered_prior_data_per_cls[ids[:count]]), 1)
        return result
