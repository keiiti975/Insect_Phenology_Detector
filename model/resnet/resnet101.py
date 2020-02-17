import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as tv_models


class ResNet101(nn.Module):
    """
        ResNet101
    """
    def __init__(self, n_class, use_DCL=False, use_SPN=False, division_number=7, pretrain=False, freeze=False, training=True, vis_feature=False):
        super(ResNet101, self).__init__()
        resnet = tv_models.resnet101(pretrained=pretrain)
        if freeze is True:
            for param in resnet.parameters():
                param.requires_grad = False
        self.n_class = n_class
        self.use_DCL = use_DCL
        self.use_SPN = use_SPN
        self.division_number = division_number
        self.pretrain = pretrain
        self.freeze = freeze
        self.training = training
        self.vis_feature = vis_feature
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.linear = nn.Linear(2048, n_class)
        if use_DCL is True:
            self.discriminator = Discriminator()
            self.RAN = Region_Alignment_Network(division_number)
        if use_SPN is True:
            self.SPN = Semantic_Prediction_Network()
    
    def forward(self, x):
        if self.vis_feature is True:
            model_features = {}
            for i in range(len(self.resnet)):
                if i in [4, 5, 6, 7]:
                    model_features.update({"conv_block_"+str(i-3): self.resnet[:i+1](x)})
            return model_features
        else:
            x = self.resnet(x)
            if self.use_DCL is True and self.training is True:
                predict_loc = self.RAN(x)
            x = self.avgpool(x).squeeze()
            if self.use_DCL is True and self.training is True:
                dest_or_not = self.discriminator(x)
            elif self.use_SPN is True and self.training is True:
                semantic_vector = self.SPN(x)
            x = self.linear(x)
            if self.use_DCL is True and self.training is True:
                return x, predict_loc, dest_or_not
            elif self.use_SPN is True and self.training is True:
                return x, semantic_vector
            else:
                return x

    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.linear(x)
        return x


class Region_Alignment_Network(nn.Module):
    def __init__(self, division_number=7):
        super(Region_Alignment_Network, self).__init__()
        self.division_number = division_number
        self.conv2d = nn.Conv2d(2048, 1, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d(division_number)

    def forward(self, x):
        x = F.relu(self.conv2d(x))
        x = self.avgpool(x)
        x = torch.squeeze(x).view(-1, self.division_number**2)
        return x
    

class Semantic_Prediction_Network(nn.Module):
    def __init__(self, semantic_length=100):
        super(Semantic_Prediction_Network, self).__init__()
        self.linear = nn.Linear(2048, semantic_length)

    def forward(self, x):
        x = self.linear(x)
        return x