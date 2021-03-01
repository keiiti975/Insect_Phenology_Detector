# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet.resnet_base import BasicBlock, Bottleneck, _resnet


class ResNet(nn.Module):
    """
        ResNet
    """
    
    def __init__(self, model_name, n_class, pretrain=False, param_freeze=False, vis_feature=False, use_dropout=False, activation_function="ReLU", decoder=None):
        """
            初期化関数
            引数:
                - model_name: str, ResNetの種類を指定,
                ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]の中から一つ
                - n_class: int, 分類するクラス数
                - pretrain: bool, ImageNetの転移学習を行うか否か
                - param_freeze: bool, 特徴抽出モデルの重みを固定するか否か
                - vis_feature: bool, 特徴量の可視化を行うか否か
                - use_dropout: bool, ヘッドの中にDropoutを組み込むか否か
                - activation_function: str, 発火関数を指定
                ["ReLU", "LeakyReLU", "RReLU"]の中から
                (コメント: これ以外の発火関数はうまくいかなかったため取り除いている)
                - decoder: str or None: デコーダーの構造を指定
                [None, "Concatenate", "FPN"]の中から
                Concatenateは畳み込みブロック特徴量1~4をconcatするもの
                FPNはFeature Pyramid Networkを実装したもの
        """
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
        self.resnet = nn.Sequential(*list(resnet.children())[:-2]) # エンコーダ部
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        # TrueでConcatenate中でconcatするブロックを指定できる
        # Falseは全ブロックを指定する
        self.use_conv_compression = True
        print("conv_compression == " + str(self.use_conv_compression))
        
        # デコーダ部
        if decoder == "Concatenate":
            self.adaptive_avgpool = nn.AdaptiveAvgPool2d(7)
            
            print("decoder == Concatenate")
            self.concat_layer = [True, True, True, True] # concatするブロックを指定
            print("concat_layer == " + str(self.concat_layer))
            if model_name == 'resnet18' or model_name == 'resnet34':
                layer_feature = np.array([64, 128, 256, 512])
            elif model_name == 'resnet50' or model_name == 'resnet101' or model_name == 'resnet152':
                layer_feature = np.array([256, 512, 1024, 2048])
            if self.use_conv_compression is True:
                # 出力する特徴量の大きさを揃えるための畳み込み層
                self.conv_compression = nn.Conv2d(layer_feature[self.concat_layer].sum(), layer_feature[self.concat_layer].sum(), kernel_size=1)
        elif decoder == "FPN":
            self.adaptive_avgpool = nn.AdaptiveAvgPool2d(7)
            
            print("decoder == FPN")
            if model_name == 'resnet18' or model_name == 'resnet34':
                # Top Layer
                self.toplayer = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0) # Reduce channels
                # Smooth Layers
                self.smooth1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
                self.smooth2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
                self.smooth3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
                # Lateral Layers
                self.latlayer1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
                self.latlayer2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
                self.latlayer3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
            elif model_name == 'resnet50' or model_name == 'resnet101' or model_name == 'resnet152':
                # Top Layer
                self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0) # Reduce channels
                # Smooth Layers
                self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
                self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
                self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
                # Lateral Layers
                self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
                self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
                self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        
        # ヘッドに入力する特徴量の大きさを取得
        if model_name == 'resnet18' or model_name == 'resnet34':
            if decoder == "Concatenate": linear_feature = layer_feature[self.concat_layer].sum()
            elif decoder == "FPN": linear_feature = 256
            else: linear_feature = 512
        elif model_name == 'resnet50' or model_name == 'resnet101' or model_name == 'resnet152':
            if decoder == "Concatenate": linear_feature = layer_feature[self.concat_layer].sum()
            elif decoder == "FPN": linear_feature = 1024
            else: linear_feature = 2048
               
        # ヘッド
        if use_dropout is True:
            print("use_dropout == True")
            self.linear = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(linear_feature, n_class),
            )
        else:
            print("use_dropout == False")
            self.linear = nn.Linear(linear_feature, n_class)
        
    def _upsample_add(self, x, y):
        """
            特徴量補完を行い、特徴量を足し合わせる
            FPNの中でのみ使用
        """
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False) + y
    
    def forward(self, x):
        """
            順伝播
            引数:
                - x: torch.Tensor, size==[bs, channel, H, W], モデル入力
        """
        if self.vis_feature is True:
            # 特徴量可視化のために、特徴量を取得
            # ブロック1~4の特徴量を辞書にして取得する
            model_features = {}
            for i in range(len(self.resnet)):
                if i in [4, 5, 6, 7]:
                    model_features.update({"conv_block_"+str(i-3): self.resnet[:i+1](x)})
            return model_features
        else:
            # 順伝播を行う
            if self.decoder == "Concatenate" or self.decoder == "FPN":
                output_conv1, output_conv2, output_conv3, output_conv4 = self.forward_encoder(x)
                x = self.forward_decoder(output_conv1, output_conv2, output_conv3, output_conv4)
            else:
                x = self.forward_encoder(x)
            x = self.avgpool(x).squeeze()
            x = self.linear(x)
            return x
    
    def forward_encoder(self, x):
        """
            エンコーダ部の順伝播
            引数:
                - x: torch.Tensor, size==[bs, channel, H, W], モデル入力
        """
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
        """
            デコーダ部の順伝播
            decoderが"Concatenate"もしくは"FPN"の時のみ使用
            引数:
                - output_conv1: torch.Tensor, size==[bs, channel, H, W], ブロック1の特徴量
                - output_conv2: torch.Tensor, size==[bs, channel, H, W], ブロック2の特徴量
                - output_conv3: torch.Tensor, size==[bs, channel, H, W], ブロック3の特徴量
                - output_conv4: torch.Tensor, size==[bs, channel, H, W], ブロック4の特徴量
        """
        if self.decoder == "Concatenate":
            concat_feature = []
            if self.concat_layer[0] is True:
                output_conv1 = self.adaptive_avgpool(output_conv1)
                concat_feature.append(output_conv1)
            if self.concat_layer[1] is True:
                output_conv2 = self.adaptive_avgpool(output_conv2)
                concat_feature.append(output_conv2)
            if self.concat_layer[2] is True:
                output_conv3 = self.adaptive_avgpool(output_conv3)
                concat_feature.append(output_conv3)
            if self.concat_layer[3] is True:
                concat_feature.append(output_conv4)
            output_feature = torch.cat(concat_feature, 1)
            if self.use_conv_compression is True:
                # 出力する特徴量の大きさを揃える必要がある
                x = self.relu(self.conv_compression(output_feature))
            else:
                x = output_feature
            return x
        else:
            # Top-down
            up_output_conv4 = self.relu(self.toplayer(output_conv4))
            up_output_conv3 = self._upsample_add(up_output_conv4, self.relu(self.latlayer1(output_conv3)))
            up_output_conv3 = self.relu(self.smooth1(up_output_conv3))
            up_output_conv2 = self._upsample_add(up_output_conv3, self.relu(self.latlayer2(output_conv2)))
            up_output_conv2 = self.relu(self.smooth2(up_output_conv2))
            up_output_conv1 = self._upsample_add(up_output_conv2, self.relu(self.latlayer3(output_conv1)))
            up_output_conv1 = self.relu(self.smooth3(up_output_conv1))
            # fit size
            up_output_conv3 = self.adaptive_avgpool(up_output_conv3)
            up_output_conv2 = self.adaptive_avgpool(up_output_conv2)
            up_output_conv1 = self.adaptive_avgpool(up_output_conv1)
            # classify
            x = torch.cat((up_output_conv4, up_output_conv3, up_output_conv2, up_output_conv1), 1)
            return x