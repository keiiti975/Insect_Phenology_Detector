def get_feature_sizes(input_size, tcb_layer_num, use_extra_layer):
    """
        get feature sizes of vgg
        Args:
            - input_size: int, image size, choice [320, 512, 1024]
            - tcb_layer_num: int, number of TCB blocks, choice [4, 5, 6]
            - use_extra_layer: bool, add extra layer to vgg or not
    """
    if tcb_layer_num == 4 and use_extra_layer == True:
        return [int(input_size / 8), int(input_size / 16), int(input_size / 32), int(input_size / 64)]
    elif tcb_layer_num == 4 and use_extra_layer == False:
        return [int(input_size / 4), int(input_size / 8), int(input_size / 16), int(input_size / 32)]
    elif tcb_layer_num == 5 and use_extra_layer == True:
        return [int(input_size / 4), int(input_size / 8), int(input_size / 16), int(input_size / 32), int(input_size / 64)]
    elif tcb_layer_num == 5 and use_extra_layer == False:
        return [int(input_size / 2), int(input_size / 4), int(input_size / 8), int(input_size / 16), int(input_size / 32)]
    elif tcb_layer_num == 6 and use_extra_layer == True:
        return [int(input_size / 2), int(input_size / 4), int(input_size / 8), int(input_size / 16), int(input_size / 32), int(input_size / 64)]
    else:
        print("ERROR: selected config can not get feature sizes")
