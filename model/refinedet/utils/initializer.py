import torch
import torch.nn as nn
import torch.nn.init as init


def xavier(param):
    """
        initialize with uniform distribution
        - param: layer parameter
    """
    init.xavier_uniform_(param)


def weights_init(m):
    """
        initialize layer weight
        - m: layer
    """
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def initialize_model(model, basenet_path, pretrain=False, freeze=False):
    """
        initialize model
        - model: pytorch model architecture
        - basenet_path: str
        - pretrain: bool
        - freeze: bool
    """
    print('Initializing weights ...')
    if pretrain is True:
        vgg_weights = torch.load(basenet_path)
        model.vgg.load_state_dict(vgg_weights)
        if freeze is True:
            for param in model.vgg.parameters():
                param.requires_grad = False
    else:
        model.vgg.apply(weights_init)
    # initialize newly added layers' weights with xavier method
    # refinedet_net.extras.apply(weights_init)
    model.arm_loc.apply(weights_init)
    model.arm_conf.apply(weights_init)
    model.odm_loc.apply(weights_init)
    model.odm_conf.apply(weights_init)
    model.tcb0.apply(weights_init)
    model.tcb1.apply(weights_init)
    model.tcb2.apply(weights_init)
