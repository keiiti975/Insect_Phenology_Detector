from torch import nn
import torchvision.models as tv_models


class ResNet18(nn.Module):
    """
        ResNet18
    """
    def __init__(self, n_class, pretrain=False):
        super(ResNet18, self).__init__()
        resnet = tv_models.resnet18(pretrained=pretrain)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.linear = nn.Linear(512, n_class)

    def forward(self, x):
        x = self.resnet(x)
        x = self.avgpool(x).squeeze()
        x = self.linear(x)
        return x