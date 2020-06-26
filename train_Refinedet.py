#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[ ]:


import numpy as np
from numpy.random import *
from os.path import join as pj
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import visdom

# Logger
from IO.logger import Logger
# Optimizer
from model.optimizer import AdamW
# Dataset
from dataset.detection.dataset import insects_dataset_from_voc_style_txt, collate_fn
# Loss Function
from model.refinedet.loss.multiboxloss import RefineDetMultiBoxLoss
# Model initializer
from model.refinedet.refinedet import RefineDet
# Predict
from model.refinedet.utils.predict import test_prediction
# Evaluate
from evaluation.detection.evaluate import Voc_Evaluater


# # Train Config

# In[ ]:


class args:
    # experiment name
    experiment_name = "crop_b4_2_4_8_16_32_im512_GN_WS_aaaaa"
    # paths
    data_root = "/home/tanida/workspace/Insect_Phenology_Detector/data"
    train_image_root = "/home/tanida/workspace/Insect_Phenology_Detector/data/train_refined_images"
    train_target_root = "/home/tanida/workspace/Insect_Phenology_Detector/data/train_detection_data/refinedet_all"
    test_image_root = "/home/tanida/workspace/Insect_Phenology_Detector/data/test_refined_images"
    test_target_root = "/home/tanida/workspace/Insect_Phenology_Detector/data/test_detection_data/refinedet_all"
    model_root = pj("/home/tanida/workspace/Insect_Phenology_Detector/output_model/detection/RefineDet", experiment_name)
    prc_root = pj("/home/tanida/workspace/Insect_Phenology_Detector/output_model/detection/RefineDet", experiment_name)
    # training config
    input_size = 512 # choices=[320, 512, 1024]
    crop_num = (5, 5)
    batch_size = 2
    num_workers = 2
    lr = 1e-4
    lamda = 1e-4
    tcb_layer_num = 5
    use_extra_layer = False
    max_epoch = 100
    valid_interval = 2
    save_interval = 20
    pretrain = True
    freeze = False
    optimizer = "AdamW"
    activation_function = "ReLU"
    init_function = "xavier_uniform_"
    use_CSL = False
    CSL_weight = [0.8, 1.2]
    use_GN_WS = True
    # visualization
    visdom = True
    visdom_port = 8097


# # Set cuda

# In[ ]:


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


# # Set Visdom

# In[ ]:


if args.visdom:
    # Create visdom
    vis = visdom.Visdom(port=args.visdom_port)
    
    """train_lossl"""
    win_arm_loc = vis.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(
            title='arm_loc_loss',
            xlabel='epoch',
            ylabel='loss',
            width=800,
            height=400
        )
    )
    win_arm_conf = vis.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(
            title='arm_conf_loss',
            xlabel='epoch',
            ylabel='loss',
            width=800,
            height=400
        )
    )
    win_odm_loc = vis.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(
            title='odm_loc_loss',
            xlabel='epoch',
            ylabel='loss',
            width=800,
            height=400
        )
    )
    win_odm_conf = vis.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(
            title='odm_conf_loss',
            xlabel='epoch',
            ylabel='loss',
            width=800,
            height=400
        )
    )
    win_norm_loss = vis.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(
            title='normalization_loss',
            xlabel='epoch',
            ylabel='loss',
            width=800,
            height=400
        )
    )
    win_all_loss = vis.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(
            title='train_loss',
            xlabel='epoch',
            ylabel='loss',
            width=800,
            height=400
        )
    )
    win_train_acc = vis.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(
            title='train_accuracy',
            xlabel='epoch',
            ylabel='average precision',
            width=800,
            height=400
        )
    )
    win_test_acc = vis.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(
            title='test_accuracy',
            xlabel='epoch',
            ylabel='average precision',
            width=800,
            height=400
        )
    )


# In[ ]:


def visualize(phase, visualized_data, window):
    vis.line(
        X=np.array([phase]),
        Y=np.array([visualized_data]),
        update='append',
        win=window
    )


# # Train and Test

# In[ ]:


arm_criterion = RefineDetMultiBoxLoss(2, use_ARM=False, use_CSL=False, CSL_weight=args.CSL_weight)
odm_criterion = RefineDetMultiBoxLoss(2, use_ARM=True, use_CSL=False, CSL_weight=args.CSL_weight)
l2_loss = nn.MSELoss(reduction='mean').cuda()


# In[ ]:


def train_per_epoch(model, data_loader, optimizer, epoch):
    # set refinedet to train mode
    model.train()

    # create loss counters
    arm_loc_loss = 0
    arm_conf_loss = 0
    odm_loc_loss = 0
    odm_conf_loss = 0
    all_norm_loss = 0

    # train
    for images, targets, _, _, _ in tqdm(data_loader, leave=False):
        imgs = np.asarray(images[0])
        tars = targets[0]

        refined_imgs = []
        refined_tars = []
        # refine imgs, tars
        for i in range(imgs.shape[0]):
            if tars[i].size(0) > 0:
                refined_imgs.append(imgs[i])
                refined_tars.append(tars[i])
        imgs = np.asarray(refined_imgs)
        tars = refined_tars

        # define batch_num
        if (imgs.shape[0]%args.batch_size == 0):
            batch_num = int(imgs.shape[0]/args.batch_size)
        else:
            batch_num = int(imgs.shape[0]/args.batch_size) + 1

        # random sample of batch
        iter_batch = choice(range(batch_num), batch_num, replace=False)

        # train for cropped image
        for i in iter_batch:
            images = imgs[i*args.batch_size:(i+1)*args.batch_size]
            targets = tars[i*args.batch_size:(i+1)*args.batch_size]

            # set cuda
            images = torch.from_numpy(images).cuda()
            targets = [ann.cuda() for ann in targets]

            # forward
            out = model(images)

            # calculate loss
            optimizer.zero_grad()
            arm_loss_l, arm_loss_c = arm_criterion(out, targets)
            odm_loss_l, odm_loss_c = odm_criterion(out, targets)
            arm_loss = arm_loss_l + arm_loss_c
            odm_loss = odm_loss_l + odm_loss_c
            loss = arm_loss + odm_loss

            if args.lamda != 0:
                norm_loss = 0
                for param in model.parameters():
                    param_target = torch.zeros(param.size()).cuda()
                    norm_loss += l2_loss(param, param_target)

                norm_loss = norm_loss * args.lamda
                loss += norm_loss
            else:
                norm_loss = 0

            if torch.isnan(loss) == 0:
                loss.backward()
                optimizer.step()
                arm_loc_loss += arm_loss_l.item()
                arm_conf_loss += arm_loss_c.item()
                odm_loc_loss += odm_loss_l.item()
                odm_conf_loss += odm_loss_c.item()
                all_norm_loss += norm_loss.item()

    print('epoch ' + str(epoch) + ' || ARM_L Loss: %.4f ARM_C Loss: %.4f ODM_L Loss: %.4f ODM_C Loss: %.4f NORM Loss: %.4f ||'     % (arm_loc_loss, arm_conf_loss, odm_loc_loss, odm_conf_loss, all_norm_loss))

    # visualize
    if args.visdom:
        visualize(epoch+1, arm_loc_loss, win_arm_loc)
        visualize(epoch+1, arm_conf_loss, win_arm_conf)
        visualize(epoch+1, odm_loc_loss, win_odm_loc)
        visualize(epoch+1, odm_conf_loss, win_odm_conf)
        visualize(epoch+1, all_norm_loss, win_norm_loss)
        visualize(epoch+1, arm_loc_loss + arm_conf_loss + odm_loc_loss + odm_conf_loss + all_norm_loss, win_all_loss)

def validate(evaluater, model, data_loader, crop_num, num_classes=2, nms_thresh=0.3):
    result = test_prediction(model, data_loader, crop_num, num_classes, nms_thresh)
    evaluater.set_result(result)
    eval_metrics = evaluater.get_eval_metrics()
    return eval_metrics[0]['AP']


# ### Save args

# In[ ]:


args_logger = Logger(args)
args_logger.save()


# ### Make data

# In[ ]:


print('Loading dataset for train ...')
train_dataset = insects_dataset_from_voc_style_txt(args.train_image_root, args.input_size, args.crop_num, "RefineDet", training=True, target_root=args.train_target_root)
train_data_loader = data.DataLoader(train_dataset, 1, num_workers=1, shuffle=True, collate_fn=collate_fn)
print('Loading dataset for test ...')
test_dataset = insects_dataset_from_voc_style_txt(args.test_image_root, args.input_size, args.crop_num, "RefineDet", training=False)
test_data_loader = data.DataLoader(test_dataset, 1, num_workers=1, shuffle=False, collate_fn=collate_fn)
train_valid_dataset = insects_dataset_from_voc_style_txt(args.train_image_root, args.input_size, args.crop_num, "RefineDet", training=False)
train_valid_data_loader = data.DataLoader(train_valid_dataset, 1, num_workers=1, shuffle=False, collate_fn=collate_fn)


# ### Make model

# In[ ]:


model = RefineDet(args.input_size, 2, args.tcb_layer_num, pretrain=args.pretrain, freeze=args.freeze, activation_function=args.activation_function, init_function=args.init_function, use_extra_layer=args.use_extra_layer, use_GN_WS=args.use_GN_WS)
if args.optimizer == "AdamW":
    print("optimizer = AdamW")
    optimizer = AdamW(model.parameters(), lr=args.lr)
else:
    print("optimizer = Adam")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
print(model)


# # Train

# In[ ]:


if os.path.exists(pj(args.prc_root, "train")) is False:
    os.makedirs(pj(args.prc_root, "train"))
if os.path.exists(pj(args.prc_root, "test")) is False:
    os.makedirs(pj(args.prc_root, "test"))


# In[ ]:


train_evaluater = Voc_Evaluater(args.train_image_root, args.train_target_root, pj(args.prc_root, "train"))
test_evaluater = Voc_Evaluater(args.test_image_root, args.test_target_root, pj(args.prc_root, "test"))
for epoch in range(args.max_epoch):
    train_per_epoch(model, train_data_loader, optimizer, epoch)
    
    # validate model
    if epoch != 0 and epoch % args.valid_interval == 0:
        train_ap = validate(train_evaluater, model, train_valid_data_loader, args.crop_num, num_classes=2, nms_thresh=0.3)
        test_ap = validate(test_evaluater, model, test_data_loader, args.crop_num, num_classes=2, nms_thresh=0.3)
        print("epoch: {}, train_ap={}, test_ap={}".format(epoch, train_ap, test_ap))
        if args.visdom:
            visualize(epoch+1, train_ap, win_train_acc)
            visualize(epoch+1, test_ap, win_test_acc)
    
    # save model
    if epoch != 0 and epoch % args.save_interval == 0:
        print('Saving state, epoch: ' + str(epoch))
        torch.save(model.state_dict(), args.model_root + '/RefineDet{}_{}.pth'.format(args.input_size, str(epoch)))

# final save model
print('Saving state, final')
torch.save(model.state_dict(), args.model_root + '/RefineDet{}_final.pth'.format(args.input_size))

