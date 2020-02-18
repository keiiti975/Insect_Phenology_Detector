import torch
from torch import nn
import torch.nn.functional as F
from model.resnet.resnet_base import BasicBlock, Bottleneck, _resnet


class ResNet(nn.Module):
    
    def __init__(self, model_name, n_class, pretrain=False, training=False, param_freeze=False, vis_feature=False, activation_function="ReLU"):
        super(ResNet, self).__init__()
        if model_name == 'resnet18':
            resnet = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained=pretrain, progress=True, activation_function=activation_function)
        elif model_name == 'resnet34':
            resnet = _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained=pretrain, progress=True, activation_function=activation_function)
        elif model_name == 'resnet50':
            resnet = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained=pretrain, progress=True, activation_function=activation_function)
        elif model_name == 'resnet101':
            resnet = _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained=pretrain, progress=True, activation_function=activation_function)
        elif model_name == 'resnet152':
            resnet = _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained=pretrain, progress=True, activation_function=activation_function)
            
        if param_freeze is True:
            for param in resnet.parameters():
                param.requires_grad = False
        
        self.model_name = model_name
        self.n_class = n_class
        self.pretrain = pretrain
        self.training = training
        self.param_freeze = param_freeze
        self.vis_feature = vis_feature
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        if model_name == 'resnet18' or model_name == 'resnet34':
            self.linear = nn.Linear(512, n_class)
        elif model_name == 'resnet50' or model_name == 'resnet101' or model_name == 'resnet152':
            self.linear = nn.Linear(2048, n_class)
        
    def forward(self, x):
        if self.vis_feature is True:
            model_features = {}
            for i in range(len(self.resnet)):
                if i in [4, 5, 6, 7]:
                    model_features.update({"conv_block_"+str(i-3): self.resnet[:i+1](x)})
            return model_features
        else:
            x = self.resnet(x)
            x = self.avgpool(x).squeeze()
            x = self.linear(x)
            return x