import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


class TwoMLPHead(nn.Module):
    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()
        
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.drop = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.drop(F.relu(self.fc6(x)))
        x = self.drop(F.relu(self.fc7(x)))
        return x

def make_Faster_RCNN(n_class, input_size, anchor_size, aspect_ratio, b_bone, max_insect_per_image=20, pretrain=True):
    if b_bone == "vgg16":
        b_outchannels = 512
        representation_channels = 4096
        backbone = torchvision.models.vgg16(pretrained=pretrain).features
        backbone = nn.Sequential(*list(backbone.children())[:-1])
        backbone.out_channels = b_outchannels
        anchor_generator = AnchorGenerator(sizes=anchor_size, aspect_ratios=aspect_ratio)
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)
        box_header = TwoMLPHead(b_outchannels * roi_pooler.output_size[0] ** 2, representation_channels)
        predictor = FastRCNNPredictor(representation_channels, n_class)
        model = FasterRCNN(backbone, min_size=input_size, max_size=input_size, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler
                           , box_head=box_header, box_predictor=predictor
                           , box_detections_per_img=max_insect_per_image)
    else:
        b_outchannels = 512
        representation_channels = 4096
        backbone = torchvision.models.resnet34(pretrained=pretrain)
        backbone = nn.Sequential(*list(backbone.children())[:-3])
        backbone.out_channels = b_outchannels
        anchor_generator = AnchorGenerator(sizes=anchor_size, aspect_ratios=aspect_ratio)
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)
        box_header = TwoMLPHead(b_outchannels * roi_pooler.output_size[0] ** 2, representation_channels)
        predictor = FastRCNNPredictor(representation_channels, n_class)
        model = FasterRCNN(backbone, min_size=input_size, max_size=input_size, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler
                           , box_head=box_header, box_predictor=predictor
                           , box_detections_per_img=args.max_insect_per_image)
    
    return model