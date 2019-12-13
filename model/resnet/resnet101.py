import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as tv_models


class ResNet101(nn.Module):
    """
        ResNet101
    """
    def __init__(self, n_class, use_DCL=False, division_number=7, pretrain=False, training=True):
        super(ResNet101, self).__init__()
        resnet = tv_models.resnet101(pretrained=pretrain)
        self.n_class = n_class
        self.use_DCL = use_DCL
        self.division_number = division_number
        self.pretrain = pretrain
        self.training = training
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.linear = nn.Linear(2048, n_class)
        if use_DCL is True:
            self.discriminator = Discriminator()
            self.RAN = Region_Alignment_Network(division_number)
    
    def forward(self, x):
        x = self.resnet(x)
        if self.use_DCL is True and self.training is True:
            predict_loc = self.RAN(x)
        x = self.avgpool(x).squeeze()
        if self.use_DCL is True and self.training is True:
            dest_or_not = self.discriminator(x)
        x = self.linear(x)
        if self.use_DCL is True and self.training is True:
            return x, predict_loc, dest_or_not
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