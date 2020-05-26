import torch.nn as nn
from model.refinedet.vgg_base import _vgg


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(pretrain, activation_function):
    """
        create VGG model
        Args:
            - input_channels: int, input channels for vgg
            - activation_function: str, "ReLU" or "LeakyReLU" or "RReLU"
            - batch_norm: bool, flag for using batch_norm
    """
    vgg = _vgg('vgg16', pretrain, True, activation_function)
    vgg_features = list(nn.Sequential(*list(vgg.features.children())[:-1]))
    pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    nn.init.xavier_uniform_(conv6.weight)
    nn.init.xavier_uniform_(conv7.weight)
    nn.init.constant_(conv6.bias, 0)
    nn.init.constant_(conv7.bias, 0)
    if activation_function == "ReLU":
        vgg_features += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    elif activation_function == "LeakyReLU":
        vgg_features += [pool5, conv6, nn.LeakyReLU(inplace=True), conv7, nn.LeakyReLU(inplace=True)]
    elif activation_function == "RReLU":
        vgg_features += [pool5, conv6, nn.RReLU(inplace=True), conv7, nn.RReLU(inplace=True)]
    return vgg_features


def vgg_extra():
    """
        Extra layers added to VGG for feature scaling
    """
    model_parts = [256, 'S', 512]
    layers = []
    input_channels = 1024
    flag = False
    for k, v in enumerate(model_parts):
        if input_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(input_channels, model_parts[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(input_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        input_channels = v
    return layers


def anchor_refinement_module(model_base, model_extra, vgg_source, use_extra_layer=False):
    """
        create ARM model
        Args:
            - model_base: VGG model
            - model_extra: extra layers for VGG
            - vgg_source: [int, ...], source layers of TCB
            - use_extra_layer: bool, add extra layer to vgg or not
    """
    arm_loc_layers = []
    arm_conf_layers = []
    for k, v in enumerate(vgg_source):
        arm_loc_layers += [nn.Conv2d(model_base[v].out_channels, 3 * 4, kernel_size=3, padding=1)]
        arm_conf_layers += [nn.Conv2d(model_base[v].out_channels, 3 * 2, kernel_size=3, padding=1)]
    if use_extra_layer is True:
        for k, v in enumerate(model_extra[1::2], 3):
            arm_loc_layers += [nn.Conv2d(v.out_channels, 3 * 4, kernel_size=3, padding=1)]
            arm_conf_layers += [nn.Conv2d(v.out_channels, 3 * 2, kernel_size=3, padding=1)]
    return (arm_loc_layers, arm_conf_layers)


def object_detection_module(model_base, model_extra, num_classes, vgg_source, use_extra_layer=False):
    """
        create ODM model
        Args:
            - model_base: VGG model
            - model_extra: extra layers for VGG
            - num_classes: int, number of object class
            - vgg_source: [int, ...], source layers of TCB
            - use_extra_layer: bool, add extra layer to vgg or not
    """
    odm_loc_layers = []
    odm_conf_layers = []
    for k, v in enumerate(vgg_source):
        odm_loc_layers += [nn.Conv2d(256, 3 * 4, kernel_size=3, padding=1)]
        odm_conf_layers += [nn.Conv2d(256, 3 * num_classes, kernel_size=3, padding=1)]
    if use_extra_layer is True:
        for k, v in enumerate(model_extra[1::2], 3):
            odm_loc_layers += [nn.Conv2d(256, 3 * 4, kernel_size=3, padding=1)]
            odm_conf_layers += [nn.Conv2d(256, 3 * num_classes, kernel_size=3, padding=1)]
    return (odm_loc_layers, odm_conf_layers)


def transfer_connection_blocks(tcb_source_channels, activation_function):
    """
        create TCB
        Args:
            - tcb_source_channels: [int, ...], source channels of TCB
            - relu: nn.ReLU or nn.LeakyReLU or nn.RReLU
    """
    feature_scale_layers = []
    feature_upsample_layers = []
    feature_pred_layers = []
    for k, v in enumerate(tcb_source_channels):
        if activation_function == "ReLU":
            feature_scale_layers += [nn.Conv2d(tcb_source_channels[k], 256, 3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, 3, padding=1)
            ]
            feature_pred_layers += [nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, 3, padding=1),
                                    nn.ReLU(inplace=True)
            ]
        elif activation_function == "LeakyReLU":
            feature_scale_layers += [nn.Conv2d(tcb_source_channels[k], 256, 3, padding=1),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Conv2d(256, 256, 3, padding=1)
            ]
            feature_pred_layers += [nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(256, 256, 3, padding=1),
                                    nn.LeakyReLU(inplace=True)
            ]
        elif activation_function == "RReLU":
            feature_scale_layers += [nn.Conv2d(tcb_source_channels[k], 256, 3, padding=1),
                                     nn.RReLU(inplace=True),
                                     nn.Conv2d(256, 256, 3, padding=1)
            ]
            feature_pred_layers += [nn.RReLU(inplace=True),
                                    nn.Conv2d(256, 256, 3, padding=1),
                                    nn.RReLU(inplace=True)
            ]
        if k != len(tcb_source_channels) - 1:
            feature_upsample_layers += [nn.ConvTranspose2d(256, 256, 2, 2)]
    return (feature_scale_layers, feature_upsample_layers, feature_pred_layers)
