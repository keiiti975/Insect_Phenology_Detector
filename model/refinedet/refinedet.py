import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.refinedet.utils.config import get_feature_sizes
from model.refinedet.layers.prior_box import get_prior_box
from model.refinedet.layers.detection import Detect
from model.refinedet.refinedet_base import vgg, vgg_extra, anchor_refinement_module, \
object_detection_module, transfer_connection_blocks


class RefineDet(nn.Module):

    def __init__(self, input_size, num_classes, tcb_layer_num, pretrain=False, freeze=False, activation_function="ReLU", init_function="kaiming_uniform_", use_extra_layer=False):
        """
            create RefineDet
            another function is needed to estimate output->label
            Args:
                - input_size: int, image size, choice [320, 512, 1024]
                - num_classes: int, number of object class
                - tcb_layer_num: int, number of TCB blocks, choice [4, 5, 6]
                - pretrain: bool, load pretrained vgg
                - freeze: bool, freeze vgg weight
                - activation_function: str, define activation_function
                - init_function: str, define init_function
                - use_extra_layer: bool, add extra layer to vgg or not
        """
        super(RefineDet, self).__init__()
        if input_size == 320 or input_size == 512 or input_size == 1024:
            pass
        else:
            print("ERROR: You specified size " + str(input_size) + ". However, currently only RefineDet320 and RefineDet512 and RefineDet1024 is supported!")

        if tcb_layer_num == 4:
            if use_extra_layer is True:
                vgg_source = [21, 28, -2]
                tcb_source_channels = [512, 512, 1024, 512]
            else:
                vgg_source = [14, 21, 28, -2]
                tcb_source_channels = [256, 512, 512, 1024]
        elif tcb_layer_num == 5:
            if use_extra_layer is True:
                vgg_source = [14, 21, 28, -2]
                tcb_source_channels = [256, 512, 512, 1024, 512]
            else:
                vgg_source = [7, 14, 21, 28, -2]
                tcb_source_channels = [128, 256, 512, 512, 1024]
        elif tcb_layer_num == 6:
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
        elif activation_function == "RReLU":
            print("activation_function = RReLU")
            
        if init_function == "xavier_uniform_":
            print("init_function = xavier_uniform_")
        elif init_function == "xavier_normal_":
            print("init_function = xavier_normal_")
        elif init_function == "kaiming_uniform_":
            print("init_function = kaiming_uniform_")
        elif init_function == "kaiming_normal_":
            print("init_function = kaiming_normal_")

        # config
        self.input_size = input_size
        self.num_classes = num_classes
        self.tcb_layer_num = tcb_layer_num
        self.pretrain = pretrain
        self.freeze = freeze
        self.activation_function = activation_function
        self.init_function = init_function
        self.use_extra_layer = use_extra_layer

        # compute prior anchor box
        feature_sizes = get_feature_sizes(input_size, tcb_layer_num, use_extra_layer)
        self.priors = get_prior_box(input_size, feature_sizes)

        # create models
        model_base = vgg(pretrain, activation_function)
        self.vgg = nn.ModuleList(model_base)

        model_extra = vgg_extra()
        self.vgg_extra = nn.ModuleList(model_extra)

        ARM = anchor_refinement_module(model_base, model_extra, vgg_source, use_extra_layer)
        ODM = object_detection_module(model_base, model_extra, num_classes, vgg_source, use_extra_layer)
        TCB = transfer_connection_blocks(tcb_source_channels, activation_function)

        self.arm_loc = nn.ModuleList(ARM[0])
        self.arm_conf = nn.ModuleList(ARM[1])
        self.odm_loc = nn.ModuleList(ODM[0])
        self.odm_conf = nn.ModuleList(ODM[1])
        self.tcb_feature_scale = nn.ModuleList(TCB[0])
        self.tcb_feature_upsample = nn.ModuleList(TCB[1])
        self.tcb_feature_pred = nn.ModuleList(TCB[2])
        self.init_weights()

        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect

    def forward(self, x):
        """
            forward function
            Args:
                - x: input image or batch of images. Shape: [batch, 3, size, size]
        """
        sources = list()
        tcb_source = list()
        arm_loc = list()
        arm_conf = list()
        odm_loc = list()
        odm_conf = list()

        # apply vgg up to conv4_3 relu and conv5_3 relu
        for k in range(30):
            x = self.vgg[k](x)
            if 8 == k:
                if self.use_extra_layer == True and self.tcb_layer_num == 6:
                    s = F.normalize(x, p=2, dim=1)
                    sources.append(s)
                if self.use_extra_layer == False and self.tcb_layer_num == 5:
                    s = F.normalize(x, p=2, dim=1)
                    sources.append(s)
            if 15 == k:
                if self.use_extra_layer == True and (self.tcb_layer_num == 5 or self.tcb_layer_num == 6):
                    s = F.normalize(x, p=2, dim=1)
                    sources.append(s)
                if self.use_extra_layer == False:
                    s = F.normalize(x, p=2, dim=1)
                    sources.append(s)
            if 22 == k:
                s = F.normalize(x, p=2, dim=1)
                sources.append(s)
            if 29 == k:
                s = F.normalize(x, p=2, dim=1)
                sources.append(s)

        # apply vgg up to fc7
        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        if self.use_extra_layer is True:
            for k, v in enumerate(self.vgg_extra):
                if self.activation_function == "ReLU":
                    x = F.relu(v(x), inplace=True)
                elif self.activation_function == "LeakyReLU":
                    x = F.leaky_relu(v(x), inplace=True)
                elif self.activation_function == "RReLU":
                    x = F.rrelu(v(x), inplace=True)
                if k % 2 == 1:
                    sources.append(x)

        # apply ARM to source layers
        for (x, l, c) in zip(sources, self.arm_loc, self.arm_conf):
            arm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            arm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc], 1)
        arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf], 1)

        # apply TCB to source layers
        p = None
        for k, v in enumerate(sources[::-1]):
            s = v
            # --- tcb_feature_scale ---
            for i in range(3):
                s = self.tcb_feature_scale[(self.tcb_layer_num - 1 - k) * 3 + i](s)

            # --- tcb_feature_upsample ---
            if k != 0:
                u = p
                u = self.tcb_feature_upsample[self.tcb_layer_num - 1 - k](u)
                s += u

            # --- tcb_feature_pred ---
            for i in range(3):
                s = self.tcb_feature_pred[(self.tcb_layer_num - 1 - k) * 3 + i](s)

            p = s
            tcb_source.append(s)
        tcb_source.reverse()

        # apply ODM to source layers
        for (x, l, c) in zip(tcb_source, self.odm_loc, self.odm_conf):
            odm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            odm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        odm_loc = torch.cat([o.view(o.size(0), -1) for o in odm_loc], 1)
        odm_conf = torch.cat([o.view(o.size(0), -1) for o in odm_conf], 1)

        if self.training == True:
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
                self.num_classes
            )
        return output

    def load_weights(self, base_file):
        """
            load model weight from .pth or .pkl file
            Args:
                - base_file: str, filename
        """
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def layer_initializer(self, layer):
        """
            initialize layer weight
            Args:
                - layer: layer module
        """
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
            if self.init_function == "xavier_uniform_":
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
            if self.init_function == "xavier_normal_":
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
            if self.init_function == "kaiming_uniform_":
                nn.init.kaiming_uniform_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
            if self.init_function == "kaiming_normal_":
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)

    def init_weights(self):
        """
            initialize model weight
        """
        if self.freeze is True:
            for param in self.vgg.parameters():
                param.requires_grad = False
        self.arm_loc.apply(self.layer_initializer)
        self.arm_conf.apply(self.layer_initializer)
        self.odm_loc.apply(self.layer_initializer)
        self.odm_conf.apply(self.layer_initializer)
        self.tcb_feature_scale.apply(self.layer_initializer)
        self.tcb_feature_upsample.apply(self.layer_initializer)
        self.tcb_feature_pred.apply(self.layer_initializer)
