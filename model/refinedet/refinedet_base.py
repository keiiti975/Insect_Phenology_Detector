import torch.nn as nn
from model.conv_layer import WS_Conv2d
from model.refinedet.vgg_base import _vgg


def get_vgg_with_GN(vgg_features):
    """
        create VGG with GN
        Args:
            - vgg_features: module list, pytorch vgg model
    """
    new_module_list = []
    for module in vgg_features:
        if isinstance(module, nn.Conv2d):
            new_module_list += [module, nn.GroupNorm(int(module.out_channels / 4), module.out_channels)]
        else:
            new_module_list += [module]
    return list(nn.Sequential(*new_module_list))


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(pretrain, activation_function, use_GN_WS=False):
    """
        create VGG model
        Args:
            - input_channels: int, input channels for vgg
            - activation_function: str
            - batch_norm: bool, flag for using batch_norm
    """
    if use_GN_WS is True:
        Conv2d = WS_Conv2d
    else:
        Conv2d = nn.Conv2d
    vgg = _vgg('vgg16', pretrain, True, activation_function, use_GN_WS)
    vgg_features = list(nn.Sequential(*list(vgg.features.children())[:-1]))
    pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    conv6 = Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
    conv7 = Conv2d(1024, 1024, kernel_size=1)
    nn.init.xavier_uniform_(conv6.weight)
    nn.init.xavier_uniform_(conv7.weight)
    nn.init.constant_(conv6.bias, 0)
    nn.init.constant_(conv7.bias, 0)
    if activation_function == "ReLU":
        vgg_features += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    elif activation_function == "LeakyReLU":
        vgg_features += [pool5, conv6, nn.LeakyReLU(inplace=True), conv7, nn.LeakyReLU(inplace=True)]
    elif activation_function == "ELU":
        vgg_features += [pool5, conv6, nn.ELU(inplace=True), conv7, nn.ELU(inplace=True)]
    elif activation_function == "LogSigmoid":
        vgg_features += [pool5, conv6, nn.LogSigmoid(), conv7, nn.LogSigmoid()]
    elif activation_function == "RReLU":
        vgg_features += [pool5, conv6, nn.RReLU(inplace=True), conv7, nn.RReLU(inplace=True)]
    elif activation_function == "SELU":
        vgg_features += [pool5, conv6, nn.SELU(inplace=True), conv7, nn.SELU(inplace=True)]
    elif activation_function == "CELU":
        vgg_features += [pool5, conv6, nn.CELU(inplace=True), conv7, nn.CELU(inplace=True)]
    elif activation_function == "Sigmoid":
        vgg_features += [pool5, conv6, nn.Sigmoid(), conv7, nn.Sigmoid()]
    if use_GN_WS is True:
        vgg_features = get_vgg_with_GN(vgg_features)
    return vgg_features


def vgg_extra(use_GN_WS=False):
    """
        Extra layers added to VGG for feature scaling
    """
    if use_GN_WS is True:
        Conv2d = WS_Conv2d
    else:
        Conv2d = nn.Conv2d
    model_parts = [256, 'S', 512]
    layers = []
    input_channels = 1024
    flag = False
    for k, v in enumerate(model_parts):
        if input_channels != 'S':
            if v == 'S':
                layers += [Conv2d(input_channels, model_parts[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [Conv2d(input_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        input_channels = v
    return layers


def anchor_refinement_module(model_base, model_extra, vgg_source, use_extra_layer=False, use_GN_WS=False):
    """
        create ARM model
        Args:
            - model_base: VGG model
            - model_extra: extra layers for VGG
            - vgg_source: [int, ...], source layers of TCB
            - use_extra_layer: bool, add extra layer to vgg or not
    """
    if use_GN_WS is True:
        Conv2d = WS_Conv2d
    else:
        Conv2d = nn.Conv2d
    arm_loc_layers = []
    arm_conf_layers = []
    if use_GN_WS is True:
        for k, v in enumerate(vgg_source):
            arm_loc_layers += [Conv2d(model_base[v].out_channels, 3 * 4, kernel_size=3, padding=1), nn.GroupNorm(4, 3 * 4)]
            arm_conf_layers += [Conv2d(model_base[v].out_channels, 3 * 2, kernel_size=3, padding=1), nn.GroupNorm(2, 3 * 2)]
        if use_extra_layer is True:
            for k, v in enumerate(model_extra[1::2], 3):
                arm_loc_layers += [Conv2d(v.out_channels, 3 * 4, kernel_size=3, padding=1), nn.GroupNorm(4, 3 * 4)]
                arm_conf_layers += [Conv2d(v.out_channels, 3 * 2, kernel_size=3, padding=1), nn.GroupNorm(2, 3 * 2)]
    else:
        for k, v in enumerate(vgg_source):
            arm_loc_layers += [Conv2d(model_base[v].out_channels, 3 * 4, kernel_size=3, padding=1)]
            arm_conf_layers += [Conv2d(model_base[v].out_channels, 3 * 2, kernel_size=3, padding=1)]
        if use_extra_layer is True:
            for k, v in enumerate(model_extra[1::2], 3):
                arm_loc_layers += [Conv2d(v.out_channels, 3 * 4, kernel_size=3, padding=1)]
                arm_conf_layers += [Conv2d(v.out_channels, 3 * 2, kernel_size=3, padding=1)]
    return (arm_loc_layers, arm_conf_layers)


def object_detection_module(model_base, model_extra, num_classes, vgg_source, use_extra_layer=False, use_GN_WS=False):
    """
        create ODM model
        Args:
            - model_base: VGG model
            - model_extra: extra layers for VGG
            - num_classes: int, number of object class
            - vgg_source: [int, ...], source layers of TCB
            - use_extra_layer: bool, add extra layer to vgg or not
    """
    if use_GN_WS is True:
        Conv2d = WS_Conv2d
    else:
        Conv2d = nn.Conv2d
    odm_loc_layers = []
    odm_conf_layers = []
    if use_GN_WS is True:
        for k, v in enumerate(vgg_source):
            odm_loc_layers += [Conv2d(256, 3 * 4, kernel_size=3, padding=1), nn.GroupNorm(4, 3 * 4)]
            odm_conf_layers += [Conv2d(256, 3 * num_classes, kernel_size=3, padding=1), nn.GroupNorm(num_classes, 3 * num_classes)]
        if use_extra_layer is True:
            for k, v in enumerate(model_extra[1::2], 3):
                odm_loc_layers += [Conv2d(256, 3 * 4, kernel_size=3, padding=1), nn.GroupNorm(4, 3 * 4)]
                odm_conf_layers += [Conv2d(256, 3 * num_classes, kernel_size=3, padding=1), nn.GroupNorm(num_classes, 3 * num_classes)]
    else:
        for k, v in enumerate(vgg_source):
            odm_loc_layers += [Conv2d(256, 3 * 4, kernel_size=3, padding=1)]
            odm_conf_layers += [Conv2d(256, 3 * num_classes, kernel_size=3, padding=1)]
        if use_extra_layer is True:
            for k, v in enumerate(model_extra[1::2], 3):
                odm_loc_layers += [Conv2d(256, 3 * 4, kernel_size=3, padding=1)]
                odm_conf_layers += [Conv2d(256, 3 * num_classes, kernel_size=3, padding=1)]
    return (odm_loc_layers, odm_conf_layers)


def transfer_connection_blocks(tcb_source_channels, activation_function, use_GN_WS=False):
    """
        create TCB
        Args:
            - tcb_source_channels: [int, ...], source channels of TCB
            - relu: nn.ReLU or nn.LeakyReLU or nn.RReLU
    """
    if use_GN_WS is True:
        Conv2d = WS_Conv2d
    else:
        Conv2d = nn.Conv2d
    feature_scale_layers = []
    feature_upsample_layers = []
    feature_pred_layers = []
    for k, v in enumerate(tcb_source_channels):
        if activation_function == "ReLU":
            feature_scale_layers += [Conv2d(tcb_source_channels[k], 256, 3, padding=1),
                                     nn.ReLU(inplace=True),
                                     Conv2d(256, 256, 3, padding=1)
            ]
            feature_pred_layers += [nn.ReLU(inplace=True),
                                    Conv2d(256, 256, 3, padding=1),
                                    nn.ReLU(inplace=True)
            ]
        elif activation_function == "LeakyReLU":
            feature_scale_layers += [Conv2d(tcb_source_channels[k], 256, 3, padding=1),
                                     nn.LeakyReLU(inplace=True),
                                     Conv2d(256, 256, 3, padding=1)
            ]
            feature_pred_layers += [nn.LeakyReLU(inplace=True),
                                    Conv2d(256, 256, 3, padding=1),
                                    nn.LeakyReLU(inplace=True)
            ]
        elif activation_function == "ELU":
            feature_scale_layers += [Conv2d(tcb_source_channels[k], 256, 3, padding=1),
                                     nn.ELU(inplace=True),
                                     Conv2d(256, 256, 3, padding=1)
            ]
            feature_pred_layers += [nn.ELU(inplace=True),
                                    Conv2d(256, 256, 3, padding=1),
                                    nn.ELU(inplace=True)
            ]
        elif activation_function == "LogSigmoid":
            feature_scale_layers += [Conv2d(tcb_source_channels[k], 256, 3, padding=1),
                                     nn.LogSigmoid(),
                                     Conv2d(256, 256, 3, padding=1)
            ]
            feature_pred_layers += [nn.LogSigmoid(),
                                    Conv2d(256, 256, 3, padding=1),
                                    nn.LogSigmoid()
            ]
        elif activation_function == "RReLU":
            feature_scale_layers += [Conv2d(tcb_source_channels[k], 256, 3, padding=1),
                                     nn.RReLU(inplace=True),
                                     Conv2d(256, 256, 3, padding=1)
            ]
            feature_pred_layers += [nn.RReLU(inplace=True),
                                    Conv2d(256, 256, 3, padding=1),
                                    nn.RReLU(inplace=True)
            ]
        elif activation_function == "SELU":
            feature_scale_layers += [Conv2d(tcb_source_channels[k], 256, 3, padding=1),
                                     nn.SELU(inplace=True),
                                     Conv2d(256, 256, 3, padding=1)
            ]
            feature_pred_layers += [nn.SELU(inplace=True),
                                    Conv2d(256, 256, 3, padding=1),
                                    nn.SELU(inplace=True)
            ]
        elif activation_function == "CELU":
            feature_scale_layers += [Conv2d(tcb_source_channels[k], 256, 3, padding=1),
                                     nn.CELU(inplace=True),
                                     Conv2d(256, 256, 3, padding=1)
            ]
            feature_pred_layers += [nn.CELU(inplace=True),
                                    Conv2d(256, 256, 3, padding=1),
                                    nn.CELU(inplace=True)
            ]
        elif activation_function == "Sigmoid":
            feature_scale_layers += [Conv2d(tcb_source_channels[k], 256, 3, padding=1),
                                     nn.Sigmoid(),
                                     Conv2d(256, 256, 3, padding=1)
            ]
            feature_pred_layers += [nn.Sigmoid(),
                                    Conv2d(256, 256, 3, padding=1),
                                    nn.Sigmoid()
            ]
        if k != len(tcb_source_channels) - 1:
            feature_upsample_layers += [nn.ConvTranspose2d(256, 256, 2, 2)]
    return (feature_scale_layers, feature_upsample_layers, feature_pred_layers)
