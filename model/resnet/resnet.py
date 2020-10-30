import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet.resnet_base import BasicBlock, Bottleneck, _resnet


class ResNet(nn.Module):
    
    def __init__(self, model_name, n_class, pretrain=False, param_freeze=False, vis_feature=False, use_dropout=False, activation_function="ReLU", decoder=None):
        super(ResNet, self).__init__()
        if activation_function == "ReLU":
            print("activation_function == ReLU")
            self.relu = nn.ReLU(inplace=True)
        elif activation_function == "LeakyReLU":
            print("activation_function == LeakyReLU")
            self.relu = nn.LeakyReLU(inplace=True)
        elif activation_function == "RReLU":
            print("activation_function == RReLU")
            self.relu = nn.RReLU(inplace=True)
        
        if model_name == 'resnet18':
            resnet = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained=pretrain, progress=True, activation_function=self.relu)
        elif model_name == 'resnet34':
            resnet = _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained=pretrain, progress=True, activation_function=self.relu)
        elif model_name == 'resnet50':
            resnet = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained=pretrain, progress=True, activation_function=self.relu)
        elif model_name == 'resnet101':
            resnet = _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained=pretrain, progress=True, activation_function=self.relu)
        elif model_name == 'resnet152':
            resnet = _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained=pretrain, progress=True, activation_function=self.relu)
        else:
            print("error: model_name missmatch!")
            
        if param_freeze is True:
            for param in resnet.parameters():
                param.requires_grad = False
        
        self.model_name = model_name
        self.n_class = n_class
        self.pretrain = pretrain
        self.param_freeze = param_freeze
        self.vis_feature = vis_feature
        self.use_dropout = use_dropout
        self.activation_function = activation_function
        self.decoder = decoder
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        # if decoder == None or Concatenate, kernel_size=7, if decoder == FPN, kernel_size=50
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        
        if model_name == 'resnet18' or model_name == 'resnet34':
            if use_dropout is True:
                print("use_dropout == True")
                self.linear = nn.Sequential(
                    nn.Dropout(p=0.5, inplace=True),
                    nn.Linear(512, n_class),
                )
            else:
                print("use_dropout == False")
                self.linear = nn.Linear(512, n_class)
        elif model_name == 'resnet50' or model_name == 'resnet101' or model_name == 'resnet152':
            if use_dropout is True:
                print("use_dropout == True")
                self.linear = nn.Sequential(
                    nn.Dropout(p=0.5, inplace=True),
                    nn.Linear(2048, n_class),
                )
            else:
                print("use_dropout == False")
                self.linear = nn.Linear(2048, n_class)
        
        if decoder == "Concatenate":
            print("decoder == Concatenate")
            self.adaptive_avgpool = nn.AdaptiveAvgPool2d(7)
            if model_name == 'resnet18' or model_name == 'resnet34':
                self.conv1 = nn.Conv2d(960, 512, kernel_size=1)
            elif model_name == 'resnet50' or model_name == 'resnet101' or model_name == 'resnet152':
                self.conv1 = nn.Conv2d(3840, 2048, kernel_size=1)
        elif decoder == "FPN":
            print("decoder == FPN")
            self.avgpool = nn.AvgPool2d(kernel_size=50, stride=1)
            if model_name == 'resnet18' or model_name == 'resnet34':
                self.conv1 = nn.Conv2d(576, 512, kernel_size=1)
                self.upconv2 = nn.ConvTranspose2d(640, 512, 3, stride=2, padding=1, output_padding=1)
                self.upconv3 = nn.ConvTranspose2d(768, 512, 3, stride=2, padding=1)
                self.upconv4 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1)
            elif model_name == 'resnet50' or model_name == 'resnet101' or model_name == 'resnet152':
                self.conv1 = nn.Conv2d(2304, 2048, kernel_size=1)
                self.upconv2 = nn.ConvTranspose2d(2560, 2048, 3, stride=2, padding=1, output_padding=1)
                self.upconv3 = nn.ConvTranspose2d(3072, 2048, 3, stride=2, padding=1)
                self.upconv4 = nn.ConvTranspose2d(2048, 2048, 3, stride=2, padding=1)
        
    def forward(self, x):
        if self.vis_feature is True:
            model_features = {}
            for i in range(len(self.resnet)):
                if i in [4, 5, 6, 7]:
                    model_features.update({"conv_block_"+str(i-3): self.resnet[:i+1](x)})
            return model_features
        else:
            if self.decoder == "Concatenate" or self.decoder == "FPN":
                output_conv1, output_conv2, output_conv3, output_conv4 = self.forward_encoder(x)
                x = self.forward_decoder(output_conv1, output_conv2, output_conv3, output_conv4)
            else:
                x = self.forward_encoder(x)
            x = self.avgpool(x).squeeze()
            x = self.linear(x)
            return x
    
    def forward_encoder(self, x):
        if self.decoder == "Concatenate" or self.decoder == "FPN":
            output_conv1 = self.resnet[:5](x)
            output_conv2 = self.resnet[5](output_conv1)
            output_conv3 = self.resnet[6](output_conv2)
            output_conv4 = self.resnet[7](output_conv3)
            return output_conv1, output_conv2, output_conv3, output_conv4
        else:
            x = self.resnet(x)
            return x
        
    def forward_decoder(self, output_conv1, output_conv2, output_conv3, output_conv4):
        if self.decoder == "Concatenate":
            output_conv1 = self.adaptive_avgpool(output_conv1)
            output_conv2 = self.adaptive_avgpool(output_conv2)
            output_conv3 = self.adaptive_avgpool(output_conv3)
            output_feature = torch.cat((output_conv1, output_conv2, output_conv3, output_conv4), 1)
            x = self.relu(self.conv1(output_feature))
            return x
        else:
            up_output_conv4 = self.relu(self.upconv4(output_conv4))
            output_conv3 = torch.cat((output_conv3, up_output_conv4), 1)
            up_output_conv3 = self.relu(self.upconv3(output_conv3))
            output_conv2 = torch.cat((output_conv2, up_output_conv3), 1)
            up_output_conv2 = self.relu(self.upconv2(output_conv2))
            output_conv1 = torch.cat((output_conv1, up_output_conv2), 1)
            x = self.relu(self.conv1(output_conv1))
            return x
        
    def forward_mahalanobis(self, x):
        # return resnet's final layer feature
        if self.decoder == "Concatenate" or self.decoder == "FPN":
            output_conv1, output_conv2, output_conv3, output_conv4 = self.forward_encoder(x)
            x = self.forward_decoder(output_conv1, output_conv2, output_conv3, output_conv4)
        else:
            x = self.forward_encoder(x)
        x = self.avgpool(x).squeeze()
        return x