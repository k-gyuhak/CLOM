import time
import numpy as np

import torch.optim

import models.transform_layers as TL
from training.contrastive_loss import get_similarity_matrix, Supervised_NT_xent
from utils.utils import AverageMeter, normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)

def update_s(P, b, B):
    """ b: current batch, B: total num batch """
    s = (P.smax - 1 / P.smax) * b / B + 1 / P.smax
    return s

def compensation(P, model, thres_cosh=50, s=1):
    """ Equation before Eq. (4) in HAT paper """
    for n, p in model.named_parameters():
        if 'ec' in n:
            if p.grad is not None:
                num = torch.cosh(torch.clamp(s * p.data, -thres_cosh, thres_cosh)) + 1
                den = torch.cosh(p.data) + 1
                p.grad *= P.smax / s * num / den

def compensation_clamp(model, thres_emb=6):
    # Constrain embeddings
    for n, p in model.named_parameters():
        if 'ec' in n:
            if p.grad is not None:
                p.data.copy_(torch.clamp(p.data, -thres_emb, thres_emb))

def hat_reg(P, p_mask, masks):
    """ masks and p_mask must have values in the same order """
    reg, count = 0., 0.
    if p_mask is not None:
        for m, mp in zip(masks, p_mask.values()):
            aux = 1. - mp#.to(device)
            reg += (m * aux).sum()
            count += aux.sum()
        reg /= count
        return P.lamb1 * reg
    else:
        for m in masks:
            reg += m.sum()
            count += np.prod(m.size()).item()
        reg /= count
        return P.lamb0 * reg

def train(P, epoch, model, criterion, optimizer, scheduler, loader, p_mask, mask_back, logger=None,
          simclr_aug=None, linear=None, linear_optim=None,
          thres_cosh=50, thres_emb=6):
    # train feature extractor, contrastive projection, and linear (without ensemble for reference)
    # The first optimizer optmizes feature extractor and contrastive projection
    # The second optimizer optimizes classifier given the features from feature extrctor

    enabled = False
    if P.amp:
        enabled = True
        torch.backends.cudnn.benchmark = True
        scaler = torch.cuda.amp.GradScaler(enabled=enabled)


    # currently only support rotation shifting augmentation
    assert simclr_aug is not None
    assert P.sim_lambda == 1.0

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['sim'] = AverageMeter()

    check = time.time()
    for n, (images, labels) in enumerate(loader):
        labels = labels % P.n_cls_per_task
        model.train()
        count = n * P.n_gpus  # number of trained samples

        data_time.update(time.time() - check)
        check = time.time()

        # Update degree of step
        s = update_s(P, n, len(loader))

        # Data augmentation B -> 2B
        batch_size = images.size(0)
        images = images.to(device)
        images1, images2 = hflip(images.repeat(2, 1, 1, 1)).chunk(2)

        # Data rotation 2B -> 8B
        images1 = torch.cat([torch.rot90(images1, rot, (2, 3)) for rot in range(4)])
        images2 = torch.cat([torch.rot90(images2, rot, (2, 3)) for rot in range(4)])
        images_pair = torch.cat([images1, images2], dim=0)

        # Assign each rotation a class
        labels = labels.to(device)
        rot_sim_labels = torch.cat([labels + P.n_cls_per_task * i for i in range(4)], dim=0)
        rot_sim_labels = rot_sim_labels.to(device)

        # Data augmentation (color_jitter, color_gray, resize_crop)
        images_pair = simclr_aug(images_pair)

        with torch.cuda.amp.autocast(enabled=enabled):
            _, outputs_aux, masks = model(P.t, images_pair, s=s, simclr=True, penultimate=True)

            # Compute supclr loss
            simclr = normalize(outputs_aux['simclr'])
            sim_matrix = get_similarity_matrix(simclr, multi_gpu=P.multi_gpu)
            loss_sim = Supervised_NT_xent(sim_matrix, labels=rot_sim_labels,
                                          temperature=0.07, multi_gpu=P.multi_gpu) * P.sim_lambda

            # HAT regularization
            loss = loss_sim
            loss += hat_reg(P, p_mask, masks)

        optimizer.zero_grad()

        hat = False
        if P.t > 0:
            hat = True

        if P.amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # embedding gradient compensation. Refer to HAT
        compensation(P, model, thres_cosh, s=s)

        if P.amp:
            if P.optimizer == 'lars':
                scaler.step(optimizer, hat=hat)
                scaler.update()
            else:
                raise NotImplementedError("feature training must be protected by HAT")
        else:
            if P.optimizer == 'lars':
                optimizer.step(hat=hat)
            else:
                raise NotImplementedError("feature training must be protected by HAT")

        # clamp embedding
        compensation_clamp(model, thres_emb)

        scheduler.step(epoch - 1 + n / len(loader))
        lr = optimizer.param_groups[0]['lr']

        batch_time.update(time.time() - check)

        # Train the standard classifier without ensemble for reference.
        # penul_1 is one zero rotation batch and penul_2 is another zero rotation batch
        penul_1 = outputs_aux['penultimate'][:batch_size]
        penul_2 = outputs_aux['penultimate'][4 * batch_size:5 * batch_size]
        outputs_aux['penultimate'] = torch.cat([penul_1, penul_2])

        outputs_linear_eval = linear[P.t](outputs_aux['penultimate'].detach())
        loss_linear = criterion(outputs_linear_eval, labels.repeat(2))

        linear_optim.zero_grad()
        loss_linear.backward()
        linear_optim.step()

        losses['cls'].update(0, batch_size)
        losses['sim'].update(loss_sim.item(), batch_size)

        if count % 50 == 0:
            P.logger.print('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossC %f] [LossSim %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['cls'].value, losses['sim'].value))

        check = time.time()

    P.logger.print('[DONE] [Time %.3f] [Data %.3f] [LossC %f] [LossSim %f]' %
         (batch_time.average, data_time.average,
          losses['cls'].average, losses['sim'].average))
