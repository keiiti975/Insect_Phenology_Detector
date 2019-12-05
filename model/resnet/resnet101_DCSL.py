import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as tv_models


class ResNet101_DCSL(nn.Module):
    def __init__(self, n_class, division_number, semantic_length, training=True):
        super(ResNet101_DCSL, self).__init__()
        resnet = tv_models.resnet101(pretrained=True)
        self.training = training
        self.division_number = division_number
        self.semantic_length = semantic_length
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.maxpool = nn.MaxPool2d(kernel_size=7)
        self.linear = nn.Linear(2048, n_class)
        self.discriminator = Discriminator()
        self.SPN = Semantic_Prediction_Network(semantic_length)
        self.RAN = Region_Alignment_Network(division_number)

    def forward(self, x):
        x = self.resnet(x)
        if self.training is True:
            predict_loc = self.RAN(x)
        else:
            predict_loc = None
        x = self.maxpool(x).squeeze()
        if self.training is True:
            dest_or_not = self.discriminator(x)
            predict_semantic = self.SPN(x)
        else:
            dest_or_not = None
            predict_semantic = None
        x = self.linear(x)
        return x, predict_loc, dest_or_not, predict_semantic


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.linear(x)
        return x


class Semantic_Prediction_Network(nn.Module):
    def __init__(self, semantic_length=100):
        super(Semantic_Prediction_Network, self).__init__()
        self.linear = nn.Linear(2048, semantic_length)

    def forward(self, x):
        x = self.linear(x)
        return x


class Region_Alignment_Network(nn.Module):
    def __init__(self, division_number=9):
        super(Region_Alignment_Network, self).__init__()
        self.division_number = division_number
        self.conv2d = nn.Conv2d(2048, 1, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d(division_number)

    def forward(self, x):
        x = F.relu(self.conv2d(x))
        x = self.avgpool(x)
        x = torch.squeeze(x).view(-1, self.division_number**2)
        return x

