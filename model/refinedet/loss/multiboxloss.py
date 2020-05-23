import torch
import torch.nn as nn
import torch.nn.functional as F
#from model.refinedet.utils.functions import refine_match
from model.refinedet.utils.functions import refine_match, log_sum_exp


class RefineDetMultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """
    def __init__(self, num_classes, use_ARM=False, use_CSL=False, CSL_weight=[1.2, 0.8]):
        super(RefineDetMultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.use_ARM = use_ARM
        self.use_CSL = use_CSL # Cost-Sensitive Learning
        self.CSL_weight = CSL_weight # Cost-Sensitive Learning weight
        self.overlap_thresh = 0.5
        self.use_prior_for_matching = True
        self.background_label = 0
        self.do_neg_mining = True
        self.negpos_ratio = 3
        self.neg_overlap = 0.5
        self.encode_target = False
        self.use_gpu = True
        self.theta = 0.01
        self.variance = [0.1, 0.2]


    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): (arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data, prior_data)
                arm_loc_data.shape == [batch_size, num_priors, 4]
                arm_conf_data.shape == [batch_size, num_priors, 2]
                odm_loc_data.shape == [batch_size, num_priors, 4]
                odm_conf_data.shape == [batch_size, num_priors, num_classes]
                prior_data.shape == [num_priors, 4]
            targets (tensor): [target_per_cropped_img, ...]
                target_per_cropped_img: tensor(array([[x1, y1, x2, y2, cls_lbl], ...], dtype=float32))
                x1, y1, x2, y2 -> 0 <= value <= 1, value based on cropped_image
                len(targets) == batch_size
                target_per_cropped_img.shape == [num_objects, 5]
        """
        arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data, prior_data = predictions
        if self.use_ARM is True:
            loc_data, conf_data = odm_loc_data, odm_conf_data
        else:
            loc_data, conf_data = arm_loc_data, arm_conf_data
        batch_size = loc_data.size(0)
        prior_data = prior_data[:loc_data.size(1), :]
        num_priors = (prior_data.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(batch_size, num_priors, 4)
        conf_t = torch.LongTensor(batch_size, num_priors)
        for idx in range(batch_size):
            target_boxes = targets[idx][:, :-1].detach()
            target_labels = targets[idx][:, -1].detach()
            if self.use_ARM is False:
                target_labels = target_labels >= 0
            prior_data = prior_data.detach()
            if self.use_ARM is True:
                refine_match(self.overlap_thresh, target_boxes, prior_data, self.variance, target_labels,
                             loc_t, conf_t, idx, arm_loc_data=arm_loc_data[idx].detach())
            else:
                refine_match(self.overlap_thresh, target_boxes, prior_data, self.variance, target_labels,
                             loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        loc_t.requires_grad = False
        conf_t.requires_grad = False

        if self.use_ARM:
            P = F.softmax(arm_conf_data, 2)
            arm_conf_tmp = P[:, :, 1]
            object_score_index = arm_conf_tmp <= self.theta
            pos = conf_t > 0
            pos[object_score_index.detach()] = 0
        else:
            pos = conf_t > 0

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        #loss_c = torch.logsumexp(batch_conf, 1, keepdim=True) - batch_conf.gather(1, conf_t.view(-1, 1))
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(batch_size, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        N = num_pos.detach().sum().float()

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)
                           ].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        if self.use_CSL is True:
            loss_c = F.cross_entropy(conf_p, targets_weighted, weight=torch.from_numpy(np.asarray(self.CSL_weight).astype("float32")).cuda())
        else:
            loss_c = F.cross_entropy(conf_p, targets_weighted)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        # N = num_pos.detach().sum().float()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
