import torch
import torch.nn as nn
import torch.nn.functional as F

from model.refinedet.layers import *
import os


class RefineDet(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, ARM, ODM, TCB, num_classes, cfg, tcb_layer_num):
        super(RefineDet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = cfg
        self.priorbox = PriorBox(self.cfg[str(size)])
        with torch.no_grad():
            self.priors = self.priorbox.forward()
        self.size = size
        self.tcb_layer_num = tcb_layer_num

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.conv3_3_L2Norm = L2Norm(256, 10)
        self.conv4_3_L2Norm = L2Norm(512, 10)
        self.conv5_3_L2Norm = L2Norm(512, 10)
        if tcb_layer_num == 5:
            self.conv2_3_L2Norm = L2Norm(128, 10)

        self.arm_loc = nn.ModuleList(ARM[0])
        self.arm_conf = nn.ModuleList(ARM[1])
        self.odm_loc = nn.ModuleList(ODM[0])
        self.odm_conf = nn.ModuleList(ODM[1])
        # self.tcb = nn.ModuleList(TCB)
        self.tcb0 = nn.ModuleList(TCB[0])
        self.tcb1 = nn.ModuleList(TCB[1])
        self.tcb2 = nn.ModuleList(TCB[2])
        
        """
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect_RefineDet(num_classes, self.size, 0, 1000, 0.01, 0.45, 0.01, 500, cfg)
        """
            
        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect_RefineDet(num_classes, self.size, 0, 1000, 0.01, 0.45, 0.01, 500, cfg)
        

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
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
            #print(x.shape)
            if self.tcb_layer_num == 5:
                if 8 == k:
                    s = self.conv2_3_L2Norm(x)
                    sources.append(s)
            if 15 == k:
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
            #print(x.shape)
        sources.append(x)
        #print("--- vgg ---")

        # apply ARM to source layers
        for (x, l, c) in zip(sources, self.arm_loc, self.arm_conf):
            #print(x.shape)
            arm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            arm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc], 1)
        arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf], 1)
        #print([x.size() for x in sources])
        # calculate TCB features
        #print([x.size() for x in sources])
        #print("--- arm ---")
        
        # apply TCB to source layers
        p = None
        for k, v in enumerate(sources[::-1]):
            s = v
            for i in range(3):
                s = self.tcb0[(self.tcb_layer_num - 1 - k) * 3 + i](s)
                #s = self.tcb0[(3-k)*3 + i](s)
                #print(s.shape)
            
            #print("--- tcb0 ---")
            if k != 0:
                u = p
                u = self.tcb1[self.tcb_layer_num - 1 - k](u)
                #u = self.tcb1[3-k](u)
                s += u
                #print(s.shape)
                
            #print("--- tcb1 ---")
            for i in range(3):
                s = self.tcb2[(self.tcb_layer_num - 1 - k) * 3 + i](s)
                #s = self.tcb2[(3-k)*3 + i](s)
                #print(s.shape)
                
            #print("--- tcb2 ---")
            p = s
            tcb_source.append(s)
        #print([x.size() for x in tcb_source])
        tcb_source.reverse()

        # apply ODM to source layers
        for (x, l, c) in zip(tcb_source, self.odm_loc, self.odm_conf):
            #print(x.shape)
            odm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            odm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        odm_loc = torch.cat([o.view(o.size(0), -1) for o in odm_loc], 1)
        odm_conf = torch.cat([o.view(o.size(0), -1) for o in odm_conf], 1)
        #print(arm_loc.size(), arm_conf.size(), odm_loc.size(), odm_conf.size())
        #print("--- odm ---")
        #print()

        if self.phase == "test":
            #print(loc, conf)
            output = self.detect(
                arm_loc.view(arm_loc.size(0), -1, 4),           # arm loc preds
                self.softmax(arm_conf.view(arm_conf.size(0), -1,
                             2)),                               # arm conf preds
                odm_loc.view(odm_loc.size(0), -1, 4),           # odm loc preds
                self.softmax(odm_conf.view(odm_conf.size(0), -1,
                             self.num_classes)),                # odm conf preds
                self.priors.type(type(x.detach()))                  # default boxes
            )
        else:
            output = (
                arm_loc.view(arm_loc.size(0), -1, 4),
                arm_conf.view(arm_conf.size(0), -1, 2),
                odm_loc.view(odm_loc.size(0), -1, 4),
                odm_conf.view(odm_conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def arm_multibox(vgg, cfg, tcb_layer_num):
    arm_loc_layers = []
    arm_conf_layers = []
    if tcb_layer_num == 4:
        vgg_source = [14, 21, 28, -2]
    elif tcb_layer_num == 5:
        vgg_source = [7, 14, 21, 28, -2]
    
    for k, v in enumerate(vgg_source):
        arm_loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        arm_conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * 2, kernel_size=3, padding=1)]
    return (arm_loc_layers, arm_conf_layers)


def odm_multibox(vgg, cfg, num_classes, tcb_layer_num):
    odm_loc_layers = []
    odm_conf_layers = []
    if tcb_layer_num == 4:
        vgg_source = [14, 21, 28, -2]
    elif tcb_layer_num == 5:
        vgg_source = [7, 14, 21, 28, -2]
    
    for k, v in enumerate(vgg_source):
        odm_loc_layers += [nn.Conv2d(256, cfg[k] * 4, kernel_size=3, padding=1)]
        odm_conf_layers += [nn.Conv2d(256, cfg[k] * num_classes, kernel_size=3, padding=1)]
    return (odm_loc_layers, odm_conf_layers)


def add_tcb(cfg):
    feature_scale_layers = []
    feature_upsample_layers = []
    feature_pred_layers = []
    for k, v in enumerate(cfg):
        feature_scale_layers += [nn.Conv2d(cfg[k], 256, 3, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(256, 256, 3, padding=1)
        ]
        feature_pred_layers += [nn.ReLU(inplace=True),
                                nn.Conv2d(256, 256, 3, padding=1),
                                nn.ReLU(inplace=True)
        ]
        if k != len(cfg) - 1:
            feature_upsample_layers += [nn.ConvTranspose2d(256, 256, 2, 2)]
    return (feature_scale_layers, feature_upsample_layers, feature_pred_layers)

"""
base = {
    '320': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '1024': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '320': [256, 'S', 512],
    '512': [256, 'S', 512],
    '1024': [256, 'S', 512],
}
mbox = {
    '320': [3, 3, 3, 3],  # number of boxes per feature map location
    #'512': [3, 3, 3, 3],  # number of boxes per feature map location
    '512': [3, 3, 3, 3, 3],  # number of boxes per feature map location
    #'1024': [3, 3, 3, 3],  # number of boxes per feature map location
    '1024': [3, 3, 3, 3, 3],  # number of boxes per feature map location
}

tcb = {
    '320': [512, 512, 1024, 512],
    #'512': [512, 512, 1024, 512],
    '512': [256, 512, 512, 1024, 512],
    #'1024': [512, 512, 1024, 512],
    '1024': [256, 512, 512, 1024, 512],
}
"""


def build_refinedet(phase, cfg, size=320, tcb_layer_num=4, num_classes=2):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 320 and size != 512 and size != 1024:
        print("ERROR: You specified size " + repr(cfg['min_dim']) + ". However, " +
              "currently only RefineDet320 and RefineDet512 and RefineDet1024 is supported!")
        return
    
    base = {
        '320': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
                512, 512, 512],
        '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
                512, 512, 512],
        '1024': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
                512, 512, 512],
    }
    
    if tcb_layer_num == 4:
        mbox = {
            '320': [3, 3, 3, 3],  # number of boxes per feature map location
            '512': [3, 3, 3, 3],  # number of boxes per feature map location
            '1024': [3, 3, 3, 3],  # number of boxes per feature map location
        }
        
        tcb = {
            '320': [256, 512, 512, 1024],
            '512': [256, 512, 512, 1024],
            '1024': [256, 512, 512, 1024],
    }
    elif tcb_layer_num == 5:
        mbox = {
            '320': [3, 3, 3, 3, 3],  # number of boxes per feature map location
            '512': [3, 3, 3, 3, 3],  # number of boxes per feature map location
            '1024': [3, 3, 3, 3, 3],  # number of boxes per feature map location
        }
        
        tcb = {
            '320': [128, 256, 512, 512, 1024],
            '512': [128, 256, 512, 512, 1024],
            '1024': [128, 256, 512, 512, 1024],
    }
    else:
        print("ERROR: tcb_layer_num: " + tcb_layer_num + " not recognized")
        return
        
    base_ = vgg(base[str(size)], 3)
    ARM_ = arm_multibox(base_, mbox[str(size)], tcb_layer_num)
    ODM_ = odm_multibox(base_, mbox[str(size)], num_classes, tcb_layer_num)
    TCB_ = add_tcb(tcb[str(size)])
    return RefineDet(phase, size, base_, ARM_, ODM_, TCB_, num_classes, cfg, tcb_layer_num)
