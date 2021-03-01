# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from model.refinedet.utils.config import get_feature_sizes
from model.refinedet.layers.l2norm import L2Norm
from model.refinedet.layers.detection import Detect
from model.refinedet.layers.prior_box import get_prior_box
from model.refinedet.refinedet_base import vgg, vgg_extra, anchor_refinement_module, object_detection_module, transfer_connection_blocks


class RefineDet(nn.Module):

    def __init__(self, input_size, num_classes, tcb_layer_num, pretrain=False, freeze=False, activation_function="ReLU", init_function="xavier_uniform_", use_extra_layer=False, use_GN_WS=False):
        """
            初期化関数
            引数:
                - input_size: int, refinedetの入力サイズを指定,
                [320, 512, 1024]の中から一つ
                - num_classes: int, 分類するクラス数(前景+背景)
                - tcb_layer_num: int, TCB(Transfer Connection Block)の数,
                [4, 5, 6]の中から一つ
                - pretrain: bool, ImageNetの転移学習を行うか否か
                - freeze: bool, 特徴抽出モデル(VGG)の重みを固定するか否か
                - activation_function: str, 発火関数を指定
                - init_function: str, 初期化関数を指定
                - use_extra_layer: bool, VGGに追加の層を入れるかどうか
                追加の層を入れるとvgg_sourceが一つずつずれる
                例)tcb_layer_num == 4, use_GN_WS == Falseのとき,
                use_extra_layer == False: vgg_source = [14, 21, 28, -2]
                ->                 True:  vgg_source = [21, 28, -2]
                - use_GN_WS: bool, Group Normalization + Weight Standardizationを使用するか否か
                RefineDetにはバッチ正規化が入っていないので, その代わりとなるもの
                これらを採用したのは, 単に性能が良さそうだったから
        """
        super(RefineDet, self).__init__()
        
        if input_size == 320 or input_size == 512 or input_size == 1024:
            pass
        else:
            print("ERROR: You specified size " + str(input_size) + ". However, currently only RefineDet320 and RefineDet512 and RefineDet1024 is supported!")

        if tcb_layer_num == 4:
            if use_GN_WS is True:
                if use_extra_layer is True:
                    vgg_source = [30, 40, -3]
                    tcb_source_channels = [512, 512, 1024, 512]
                else:
                    vgg_source = [20, 30, 40, -3]
                    tcb_source_channels = [256, 512, 512, 1024]
            else:
                if use_extra_layer is True:
                    vgg_source = [21, 28, -2]
                    tcb_source_channels = [512, 512, 1024, 512]
                else:
                    vgg_source = [14, 21, 28, -2]
                    tcb_source_channels = [256, 512, 512, 1024]
        elif tcb_layer_num == 5:
            if use_GN_WS is True:
                if use_extra_layer is True:
                    vgg_source = [20, 30, 40, -3]
                    tcb_source_channels = [256, 512, 512, 1024, 512]
                else:
                    vgg_source = [10, 20, 30, 40, -3]
                    tcb_source_channels = [128, 256, 512, 512, 1024]
            else:
                if use_extra_layer is True:
                    vgg_source = [14, 21, 28, -2]
                    tcb_source_channels = [256, 512, 512, 1024, 512]
                else:
                    vgg_source = [7, 14, 21, 28, -2]
                    tcb_source_channels = [128, 256, 512, 512, 1024]
        elif tcb_layer_num == 6:
            if use_GN_WS is True:
                if use_extra_layer is True:
                    vgg_source = [10, 20, 30, 40, -3]
                    tcb_source_channels = [128, 256, 512, 512, 1024, 512]
                else:
                    print("ERROR: tcb_layer_num=6 and use_extra_layer=False is not defined")
            else:
                if use_extra_layer is True:
                    vgg_source = [7, 14, 21, 28, -2]
                    tcb_source_channels = [128, 256, 512, 512, 1024, 512]
                else:
                    print("ERROR: tcb_layer_num=6 and use_extra_layer=False is not defined")
        else:
            print("ERROR: You specified tcb_layer_num " + str(tcb_layer_num) + ". 4,5,6 is allowed for this value")

        if activation_function == "ReLU":
            print("activation_function = ReLU")
        elif activation_function == "LeakyReLU":
            print("activation_function = LeakyReLU")
        elif activation_function == "ELU":
            print("activation_function = ELU")
        elif activation_function == "LogSigmoid":
            print("activation_function = LogSigmoid")
        elif activation_function == "RReLU":
            print("activation_function = RReLU")
        elif activation_function == "SELU":
            print("activation_function = SELU")
        elif activation_function == "CELU":
            print("activation_function = CELU")
        elif activation_function == "Sigmoid":
            print("activation_function = Sigmoid")
            
        if init_function == "xavier_uniform_":
            print("init_function = xavier_uniform_")
        elif init_function == "xavier_normal_":
            print("init_function = xavier_normal_")
        elif init_function == "kaiming_uniform_":
            print("init_function = kaiming_uniform_")
        elif init_function == "kaiming_normal_":
            print("init_function = kaiming_normal_")
        elif init_function == "orthogonal_":
            print("init_function = orthogonal_")

        # config
        self.input_size = input_size
        self.num_classes = num_classes
        self.tcb_layer_num = tcb_layer_num
        self.pretrain = pretrain
        self.freeze = freeze
        self.activation_function = activation_function
        self.init_function = init_function
        self.use_extra_layer = use_extra_layer
        self.use_GN_WS = use_GN_WS

        # compute prior anchor box
        feature_sizes = get_feature_sizes(input_size, tcb_layer_num, use_extra_layer)
        self.priors = get_prior_box(input_size, feature_sizes)

        # create models
        model_base = vgg(pretrain, activation_function, use_GN_WS)
        self.vgg = nn.ModuleList(model_base)

        if self.use_extra_layer is True:
            model_extra = vgg_extra(use_GN_WS)
            self.extras = nn.ModuleList(model_extra)
        else:
            model_extra = None

        ARM = anchor_refinement_module(model_base, model_extra, vgg_source, use_extra_layer, use_GN_WS)
        ODM = object_detection_module(model_base, model_extra, num_classes, vgg_source, use_extra_layer, use_GN_WS)
        TCB = transfer_connection_blocks(tcb_source_channels, activation_function, use_GN_WS)
        
        self.conv2_3_L2Norm = L2Norm(128)
        self.conv3_3_L2Norm = L2Norm(256)
        self.conv4_3_L2Norm = L2Norm(512)
        self.conv5_3_L2Norm = L2Norm(512)
        
        self.arm_loc = nn.ModuleList(ARM[0])
        self.arm_conf = nn.ModuleList(ARM[1])
        self.odm_loc = nn.ModuleList(ODM[0])
        self.odm_conf = nn.ModuleList(ODM[1])
        self.tcb0 = nn.ModuleList(TCB[0])
        self.tcb1 = nn.ModuleList(TCB[1])
        self.tcb2 = nn.ModuleList(TCB[2])
        self.init_weights()

        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect
    
    def forward(self, x):
        """
            順伝播
            引数:
                - x: torch.Tensor, size==[bs, 3, input_size, input_size], モデル入力
        """
        sources = list()
        tcb_source = list()
        arm_loc = list()
        arm_conf = list()
        odm_loc = list()
        odm_conf = list()

        if self.use_GN_WS is True:
            # apply vgg up to conv4_3 relu and conv5_3 relu
            for k in range(43):
                x = self.vgg[k](x)
                if 12 == k:
                    if self.use_extra_layer is True and self.tcb_layer_num == 6:
                        s = self.conv2_3_L2Norm(x)
                        sources.append(s)
                    if self.use_extra_layer is False and self.tcb_layer_num == 5:
                        s = self.conv2_3_L2Norm(x)
                        sources.append(s)
                if 22 == k:
                    if self.use_extra_layer is True and (self.tcb_layer_num == 5 or self.tcb_layer_num == 6):
                        s = self.conv3_3_L2Norm(x)
                        sources.append(s)
                    if self.use_extra_layer is False:
                        s = self.conv3_3_L2Norm(x)
                        sources.append(s)
                if 32 == k:
                    s = self.conv4_3_L2Norm(x)
                    sources.append(s)
                if 42 == k:
                    s = self.conv5_3_L2Norm(x)
                    sources.append(s)

            # apply vgg up to fc7
            for k in range(43, len(self.vgg)):
                x = self.vgg[k](x)
            sources.append(x)
        else:
            # apply vgg up to conv4_3 relu and conv5_3 relu
            for k in range(30):
                x = self.vgg[k](x)
                if 8 == k:
                    if self.use_extra_layer is True and self.tcb_layer_num == 6:
                        s = self.conv2_3_L2Norm(x)
                        sources.append(s)
                    if self.use_extra_layer is False and self.tcb_layer_num == 5:
                        s = self.conv2_3_L2Norm(x)
                        sources.append(s)
                if 15 == k:
                    if self.use_extra_layer is True and (self.tcb_layer_num == 5 or self.tcb_layer_num == 6):
                        s = self.conv3_3_L2Norm(x)
                        sources.append(s)
                    if self.use_extra_layer is False:
                        s = self.conv3_3_L2Norm(x)
                        sources.append(s)
                if 22 == k:
                    s = self.conv4_3_L2Norm(x)
                    sources.append(s)
                if 29 == k:
                    s = self.conv5_3_L2Norm(x)
                    sources.append(s)

            # apply vgg up to fc7
            for k in range(30, len(self.vgg)):
                x = self.vgg[k](x)
            sources.append(x)

        # apply extra layers and cache source layer outputs
        if self.use_extra_layer is True:
            for k, v in enumerate(self.extras):
                if self.activation_function == "ReLU":
                    x = F.relu(v(x), inplace=True)
                elif self.activation_function == "LeakyReLU":
                    x = F.leaky_relu(v(x), inplace=True)
                elif self.activation_function == "ELU":
                    x = F.elu(v(x), inplace=True)
                elif self.activation_function == "LogSigmoid":
                    x = F.logsigmoid(v(x))
                elif self.activation_function == "RReLU":
                    x = F.rrelu(v(x), inplace=True)
                elif self.activation_function == "SELU":
                    x = F.selu(v(x), inplace=True)
                elif self.activation_function == "CELU":
                    x = F.celu(v(x), inplace=True)
                elif self.activation_function == "Sigmoid":
                    x = F.sigmoid(v(x))
                if k % 2 == 1:
                    sources.append(x)

        # apply ARM to source layers
        if self.use_GN_WS is True:
            for k, x in enumerate(sources):
                arm_loc.append(self.arm_loc[2*k+1](self.arm_loc[2*k](x)).permute(0, 2, 3, 1).contiguous())
                arm_conf.append(self.arm_conf[2*k+1](self.arm_conf[2*k](x)).permute(0, 2, 3, 1).contiguous())
            arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc], 1)
            arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf], 1)
        else:
            for (x, l, c) in zip(sources, self.arm_loc, self.arm_conf):
                arm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                arm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc], 1)
            arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf], 1)

        # apply TCB to source layers
        p = None
        for k, v in enumerate(sources[::-1]):
            s = v
            # --- tcb0 ---
            for i in range(3):
                s = self.tcb0[(self.tcb_layer_num - 1 - k) * 3 + i](s)

            # --- tcb1 ---
            if k != 0:
                u = p
                u = self.tcb1[self.tcb_layer_num - 1 - k](u)
                s += u

            # --- tcb2 ---
            for i in range(3):
                s = self.tcb2[(self.tcb_layer_num - 1 - k) * 3 + i](s)

            p = s
            tcb_source.append(s)
        tcb_source.reverse()

        # apply ODM to source layers
        if self.use_GN_WS is True:
            for k, x in enumerate(tcb_source):
                odm_loc.append(self.odm_loc[2*k+1](self.odm_loc[2*k](x)).permute(0, 2, 3, 1).contiguous())
                odm_conf.append(self.odm_conf[2*k+1](self.odm_conf[2*k](x)).permute(0, 2, 3, 1).contiguous())
            odm_loc = torch.cat([o.view(o.size(0), -1) for o in odm_loc], 1)
            odm_conf = torch.cat([o.view(o.size(0), -1) for o in odm_conf], 1)
        else:
            for (x, l, c) in zip(tcb_source, self.odm_loc, self.odm_conf):
                odm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                odm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            odm_loc = torch.cat([o.view(o.size(0), -1) for o in odm_loc], 1)
            odm_conf = torch.cat([o.view(o.size(0), -1) for o in odm_conf], 1)

        if self.training is True:
            # if model is train mode
            output = (
                arm_loc.view(arm_loc.size(0), -1, 4),
                arm_conf.view(arm_conf.size(0), -1, 2),
                odm_loc.view(odm_loc.size(0), -1, 4),
                odm_conf.view(odm_conf.size(0), -1, self.num_classes),
                self.priors
            )
        else:
            # if model is test mode
            output = self.detect.apply(
                arm_loc.view(arm_loc.size(0), -1, 4),  # arm loc preds
                self.softmax(arm_conf.view(arm_conf.size(0), -1, 2)),  # arm conf preds
                odm_loc.view(odm_loc.size(0), -1, 4),  # odm loc preds
                self.softmax(odm_conf.view(odm_conf.size(0), -1, self.num_classes)),  # odm conf preds
                self.priors.type(type(x.detach())),  # default boxes, match type with x
                self.num_classes,
            )
        return output
    
    def init_weights(self):
        """
            重み初期化関数
        """
        print('Initializing weights ...')
        if self.freeze is True:
            for param in self.vgg.parameters():
                param.requires_grad = False
        if self.init_function == "xavier_uniform_":
            layer_initializer = xavier_uniform_initializer
        elif self.init_function == "xavier_normal_":
            layer_initializer = xavier_normal_initializer
        elif self.init_function == "kaiming_uniform_":
            layer_initializer = kaiming_uniform_initializer
        elif self.init_function == "kaiming_normal_":
            layer_initializer = kaiming_normal_initializer
        elif self.init_function == "orthogonal_":
            layer_initializer = orthogonal_initializer
        
        self.arm_loc.apply(layer_initializer)
        self.arm_conf.apply(layer_initializer)
        self.odm_loc.apply(layer_initializer)
        self.odm_conf.apply(layer_initializer)
        self.tcb0.apply(layer_initializer)
        self.tcb1.apply(layer_initializer)
        self.tcb2.apply(layer_initializer)
        
        
def xavier_uniform_initializer(layer):
    """
        initialize layer weight
        Args:
            - layer: layer module
    """
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias, 0.)

        
def xavier_normal_initializer(layer):
    """
        initialize layer weight
        Args:
            - layer: layer module
    """
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        nn.init.xavier_normal_(layer.weight)
        nn.init.constant_(layer.bias, 0.)

        
def kaiming_uniform_initializer(layer):
    """
        initialize layer weight
        Args:
            - layer: layer module
    """
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        nn.init.kaiming_uniform_(layer.weight)
        nn.init.constant_(layer.bias, 0.)

        
def kaiming_normal_initializer(layer):
    """
        initialize layer weight
        Args:
            - layer: layer module
    """
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(layer.weight)
        nn.init.constant_(layer.bias, 0.)


def orthogonal_initializer(layer):
    """
        initialize layer weight
        Args:
            - layer: layer module
    """
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        nn.init.orthogonal_(layer.weight)
        nn.init.constant_(layer.bias, 0.)