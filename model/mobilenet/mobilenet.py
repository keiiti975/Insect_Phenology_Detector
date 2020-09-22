import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as tv_models


class MobileNet(nn.Module):
    
    def __init__(self, n_class, pretrain=False, param_freeze=False):
        super(MobileNet, self).__init__()
        mobilenet = tv_models.mobilenet_v2(pretrained=pretrain, progress=True)
            
        if param_freeze is True:
            for param in mobilenet.parameters():
                param.requires_grad = False
        
        self.n_class = n_class
        self.pretrain = pretrain
        self.param_freeze = param_freeze
        self.mobilenet = nn.Sequential(*list(mobilenet.children())[:-1])
        self.dropout = nn.Dropout(p=0.2, inplace=False)
        self.linear = nn.Linear(in_features=1280, out_features=n_class, bias=True)
        
    def forward(self, x):
        x = self.mobilenet(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x