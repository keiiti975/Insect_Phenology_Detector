import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.CrossEntropyLoss):
    """
        Label Smoothing Cross Entropy Loss
    """

    def __init__(self, label_smoothing, num_classes, counts=None, knowledge=None, weight=None, size_average=None, reduction='mean'):
        """
            init function
            Args:
                - label_smoothing: float, 0~1
                - num_classes: int, insect class number
                - counts: np.array(dtype=int), shape == [num_classes], insect count
                - knowledge: np.array(dtype=float), shape == [num_classes, num_classes], softmax distribution
                - weight: torch.FloatTensor, shape == [num_classes]
                - size_average: bool
                - reduction: str, 'mean' or 'sum'
        """
        super(LabelSmoothingLoss, self).__init__(weight, size_average, reduction=reduction)
        
        # initialize label smooth loss
        assert 0.0 < label_smoothing <= 1.0
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.counts = counts
        self.knowledge = knowledge
        self.confidence = 1.0 - label_smoothing
        self.softmax = nn.Softmax(dim=1)
        self.cross_entropy = nn.CrossEntropyLoss()
        if self.counts is not None:
            one_hot = torch.Tensor(counts / counts.sum())
            self.register_buffer('one_hot', one_hot.unsqueeze(0).cuda())
        elif self.knowledge is not None:
            pass
        else:
            smoothing_value = 1.0 / num_classes
            one_hot = torch.full((num_classes, ), smoothing_value)
            self.register_buffer('one_hot', one_hot.unsqueeze(0).cuda())

    def forward(self, output, target):
        """
            forward function
            Args:
                - output: torch.FloatTensor, batchsize * num_classes
                - target: torch.Tensor, batchsize
        """
        if self.knowledge is not None:
            softmax_output = self.softmax(output)
            model_prob = torch.Tensor([self.knowledge[y] for y in target]).cuda()
            return F.kl_div(softmax_output, model_prob, reduction='batchmean')
            """
            return (1.0 - self.label_smoothing) * self.cross_entropy(output, target) + self.label_smoothing * \
                F.kl_div(softmax_output, model_prob, reduction='batchmean')
            """
        else:
            output = self.softmax(output)
            model_prob = self.one_hot.repeat(target.size(0), 1)
            model_prob = model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
            return F.kl_div(output, model_prob, reduction='batchmean')