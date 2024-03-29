import torch
import torch.nn as nn
from model.conv_layer import WS_Conv2d
from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = [
    'VGG', 'vgg16'
]


model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'
}


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, activation_function, use_GN_WS):
    if use_GN_WS is True:
        Conv2d = WS_Conv2d
    else:
        Conv2d = nn.Conv2d
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d]
            if activation_function == "ReLU":
                layers += [nn.ReLU(inplace=True)]
            elif activation_function == "LeakyReLU":
                layers += [nn.LeakyReLU(inplace=True)]
            elif activation_function == "ELU":
                layers += [nn.ELU(inplace=True)]
            elif activation_function == "LogSigmoid":
                layers += [nn.LogSigmoid()]
            elif activation_function == "RReLU":
                layers += [nn.RReLU(inplace=True)]
            elif activation_function == "SELU":
                layers += [nn.SELU(inplace=True)]
            elif activation_function == "CELU":
                layers += [nn.CELU(inplace=True)]
            elif activation_function == "Sigmoid":
                layers += [nn.Sigmoid()]
            in_channels = v
    return nn.Sequential(*layers)


def _vgg(arch, pretrained, progress, activation_function, use_GN_WS, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg, activation_function, use_GN_WS), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
