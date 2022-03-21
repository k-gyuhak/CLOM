import sys
import time

import torch.optim
import torch.optim.lr_scheduler as lr_scheduler

import models.transform_layers as TL
from utils.utils import AverageMeter, normalize

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)

def train(P, epoch, model, criterion, optimizer, scheduler, loader, p_mask, mask_back, logger=None,
          simclr_aug=None, linear=None, linear_optim=None):
    # Given a fixed feature extractor, train joint_linear (ensemble classifier)

    if P.multi_gpu:
        joint_linear = model.module.joint_distribution_layer
    else:
        joint_linear = model.joint_distribution_layer

    if epoch == 1:
        # define optimizer and save in P (argument)
        milestones = [int(0.6 * P.epochs), int(0.75 * P.epochs), int(0.9 * P.epochs)]

        joint_linear_optim = torch.optim.SGD(joint_linear.parameters(),
                                             lr=1e-1, weight_decay=P.weight_decay)
        P.joint_linear_optim = joint_linear_optim
        P.joint_scheduler = lr_scheduler.MultiStepLR(P.joint_linear_optim, gamma=0.1, milestones=milestones)

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['jnt'] = AverageMeter()

    check = time.time()
    for n, (images, labels) in enumerate(loader):
        labels = labels % P.n_cls_per_task
        model.eval()
        count = n * P.n_gpus  # number of trained samples

        data_time.update(time.time() - check)
        check = time.time()

        # horizontal flip
        batch_size = images.size(0)
        images = images.to(device)
        images = hflip(images)

        # rotation B -> 4B
        labels = labels.to(device)
        images = torch.cat([torch.rot90(images, rot, (2, 3)) for rot in range(4)])

        # Assign each rotation degree a different class
        joint_labels = torch.cat([labels + P.n_cls_per_task * i for i in range(4)], dim=0)

        # augmentation (color_jitter, color_gray, resize_crop)
        images = simclr_aug(images) 

        # Obtain features from fixed feature extractor
        with torch.no_grad():
            _, outputs_aux, masks = model(P.t, images, s=P.smax, penultimate=True)
        penultimate = outputs_aux['penultimate'].detach()

        # obtain outputs
        outputs_joint = joint_linear[P.t](penultimate)

        loss_joint = criterion(outputs_joint, joint_labels)

        P.joint_linear_optim.zero_grad()
        loss_joint.backward()
        P.joint_linear_optim.step()

        lr = P.joint_linear_optim.param_groups[0]['lr']

        batch_time.update(time.time() - check)

        losses['jnt'].update(loss_joint.item(), batch_size)

        if count % 50 == 0:
            P.logger.print('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossJ %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['jnt'].value))
        check = time.time()

    P.joint_scheduler.step()

    P.logger.print('[DONE] [Time %.3f] [Data %.3f] [LossJ %f]' %
         (batch_time.average, data_time.average,
          losses['jnt'].average))
