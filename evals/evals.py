import os
import sys
import time
import itertools

import diffdist.functional as distops
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import models.transform_layers as TL
from utils.temperature_scaling import _ECELoss
from utils.utils import AverageMeter, set_random_seed, normalize, md, auc

from datetime import datetime
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ece_criterion = _ECELoss().to(device)

def cil_pre(P, model, loaders, steps, criterion, test_loaders, marginal=False, logger=None):
    """ training output calibration """
    counter = 0

    # Initialize scaling and shifting parameters
    w = torch.rand(P.cil_task + 1, requires_grad=True, device=device)
    b = torch.rand(P.cil_task + 1, requires_grad=True, device=device)

    lr = P.adaptation_lr
    optimizer = torch.optim.SGD([w, b], lr=lr, momentum=0.8)
    mode = model.training
    model.eval()

    P.logger.print("lr = ", lr, " num epoch ", 5 * len(loaders[0]))

    for epoch in range(160 * len(loaders[0])):
        output_list, label_list = [], []

        # For each task loader
        for t_loader, loader in loaders.items():
            # load a batch from a task loader
            images, labels = iter(loader).next()
            images, labels = images.to(device), labels.to(device)

            # Obtain the output heads of the batch and concatenate them for CIL
            cil_outputs = torch.tensor([]).to(device)
            for t in range(P.cil_task + 1):
                # For ensemble
                outputs = 0
                for i in range(4):
                    with torch.no_grad():
                        rot_images = torch.rot90(images, i, (2, 3))
                        _, outputs_aux, _ = model(t, rot_images, s=P.smax, joint=True)
                        outputs += outputs_aux['joint'][:, P.n_cls_per_task * i: P.n_cls_per_task * (i + 1)] / 4.

                outputs = outputs * w[t] + b[t]
                cil_outputs = torch.cat((cil_outputs, outputs), dim=1)

            output_list.append(cil_outputs)
            label_list.append(labels)

        output_list = torch.cat(output_list)
        label_list = torch.cat(label_list)

        loss = criterion(output_list, label_list)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        now = datetime.now()
        P.logger.print(now.strftime("%m/%d/%Y %H:%M:%S"), end=' | ') 
        P.logger.print(loss.item())
        if epoch % 40 == 0:
            P.logger.print(epoch, loss.item())
            error = 0
            for t, loader in test_loaders.items():
                P.logger.print(t)
                with torch.no_grad():
                    error_task = cil(P, model, {t: loader}, 0, logger=None, w=w, b=b)
                error += error_task
                # cl_outputs['acc'][counter][t, P.cil_task] = error_task
            P.logger.print("avg error ", error / (P.cil_task + 1), " avg acc ", 100 - error / (P.cil_task + 1))
            P.logger.print(w)
            P.logger.print(b)

            torch.save({'w': w, 'b': b}, P.logout + '/calibration')

            counter += 1

            now = datetime.now()
            P.logger.print(now.strftime("%m/%d/%Y %H:%M:%S"))

        # Stop training after 160 iterations
        if epoch > 40 * 4 + 1:
            break
        elif counter == 4:
            # torch.save(cl_outputs, './' + P.logout + '/cl_outputs_adapt_w_b')
            torch.save({'w': w, 'b': b}, P.logout + '/calibration')
            break

def cil(P, model, loaders, steps, marginal=False, logger=None, T=None, w=None, b=None):
    """ 
        This is for testing CIL.
        If w and b are provided, it's CLOM. Otherwise, it's CLOM(-c) without calibration (memory free).
    """
    # Switch to evaluate mode
    mode = model.training
    model.eval()

    scores_dict = {}
    outputs_tasks, targets_tasks = [], []
    for data_id, loader in loaders.items():
        error_top1 = AverageMeter()
        error_calibration = AverageMeter()
        outputs_all, targets_all = [], []
        scores_all = []
        for n, (images, labels) in enumerate(loader): # loader is in_loader
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)

            # For a batch, obtain outputs from each task networks and concatenate for cil
            cil_outputs = torch.tensor([]).to(device)
            scores_batch = []
            for t in range(P.cil_task + 1):
                # For ensemble prediction
                outputs = 0
                for i in range(4):
                    rot_images = torch.rot90(images, i, (2, 3))
                    with torch.no_grad():
                        _, outputs_aux, _ = model(t, rot_images, s=P.smax, joint=True, penultimate=True)
                    output_ = outputs_aux['joint'][:, P.n_cls_per_task * i: P.n_cls_per_task * (i + 1)] / 4.
                    outputs += output_

                # Apply calibration if available
                if w is not None:
                    new_outputs = outputs * w[t] + b[t]
                    cil_outputs = torch.cat((cil_outputs, new_outputs.detach()), dim=1)
                else:
                    new_outputs = outputs
                    cil_outputs = torch.cat((cil_outputs, new_outputs.detach()), dim=1)

                scores, _ = torch.max(new_outputs, dim=1, keepdim=True)
                scores_batch.append(scores)

            # Top 1 error. Accuracy is 100 - error.
            top1, = error_k(cil_outputs.data, labels, ks=(1,))
            error_top1.update(top1.item(), batch_size)

            # Just for reference. Not very informative for cil
            ece = ece_criterion(cil_outputs, labels) * 100
            error_calibration.update(ece.item(), batch_size)

            outputs_all.append(cil_outputs.data.cpu().numpy())
            targets_all.append(labels.data.cpu().numpy())

            scores_batch = torch.cat(scores_batch, dim=1)
            scores_all.append(scores_batch.cpu().numpy())

            if n % 100 == 0:
                P.logger.print('[Test %3d] [Test@1 %.3f] [100-ECE %.3f]' %
                     (n, 100-error_top1.value, 100-error_calibration.value))

        P.logger.print('[Data id %3d] [ACC@1 %.3f] [100-ECE %.3f]' %
             (data_id, 100-error_top1.average, 100-error_calibration.average))
        if P.mode == 'cil':
            P.cil_tracker.update(100 - error_top1.average, P.cil_task, data_id)
        elif P.mode == 'cil_pre':
            P.cal_cil_tracker.update(100 - error_top1.average, P.cil_task, data_id)
        else:
            raise NotImplementedError()

        outputs_all = np.concatenate(outputs_all)
        targets_all = np.concatenate(targets_all)
        scores_all  = np.concatenate(scores_all)

        scores_dict[data_id] = scores_all
        outputs_tasks.append(outputs_all)
        targets_tasks.append(targets_all)

    if len(loaders) == P.cil_task + 1:
        torch.save([outputs_tasks, targets_tasks], f'{P.logout}/outputs_labels_list_{data_id}')
        for data_id in range(P.cil_task + 1):
            auc(scores_dict, data_id, P.auc_tracker)

        P.logger.print("Softmax AUC result")
        P.auc_tracker.print_result(len(loaders) - 1, type='acc')

    model.train(mode)
    return error_top1.average

def error_k(output, target, ks=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(ks)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in ks:
        correct_k = correct[:k].view(-1).float().sum(0)
        results.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return results

def test_classifier(P, model, loaders, steps, marginal=False, logger=None):
    """ 
        This is for TIL prediction
        Note. TIL must use the task network corresponding to the loaded dataset
    """
    # Switch to evaluate mode
    mode = model.training
    model.eval()

    for data_id, loader in loaders.items():
        error_top1 = AverageMeter()
        error_calibration = AverageMeter()
        for n, (images, labels) in enumerate(loader):
            labels = labels % P.n_cls_per_task
            batch_size = images.size(0)

            images, labels = images.to(device), labels.to(device)

            # marginal is True during testing. Use ensemble output.
            if marginal:
                # Ensemble
                outputs = 0
                for i in range(4):
                    rot_images = torch.rot90(images, i, (2, 3))
                    _, outputs_aux, _ = model(data_id, rot_images, s=P.smax, joint=True, penultimate=True)
                    outputs += outputs_aux['joint'][:, P.n_cls_per_task * i: P.n_cls_per_task * (i + 1)] / 4.
            # marginal is False during training. Use zero-rotation.
            else:
                outputs, _ = model(data_id, images, s=P.smax)

            # Top 1 error. Acc is 100 - error
            top1, = error_k(outputs.data, labels, ks=(1,))
            error_top1.update(top1.item(), batch_size)

            # For reference.
            ece = ece_criterion(outputs, labels) * 100
            error_calibration.update(ece.item(), batch_size)

            # Accuracy at 100th batch just for reference
            if n % 100 == 0:
                P.logger.print('[Test %3d] [Test@1 %.3f] [100-ECE %.3f]' %
                     (n, 100-error_top1.value, 100-error_calibration.value))

        # Accuracy over entire batch (i.e. the final accuracy)
        P.logger.print(' * [ACC@1 %.3f] [100-ECE %.3f]' %
             (100-error_top1.average, 100-error_calibration.average))

        # Record the test accuracy
        if marginal:
            P.til_tracker.update(100 - error_top1.average,
                                int(P.logout.split('task_')[-1]),
                                data_id)

    model.train(mode)

    return error_top1.average

def eval_ood_detection(P, model, id_loaders, ood_loaders, ood_scores, train_loader=None, simclr_aug=None):
    """
        Compute AUC of task network k. Dataset of task k is IND and all other data of task j != k are OOD
        Note. Must use task network corresponding to the loaded network (IND)
    """
    auroc_dict = dict()
    for ood in ood_loaders.keys():
        auroc_dict[ood] = dict()

    # This implementation only works for ood_scores = ['baseline_marginalized']
    for ood_score in ood_scores:
        # compute scores for ID and OOD samples
        score_func = get_ood_score_func(P, model, ood_score, simclr_aug=simclr_aug)

        save_path = f'plot/score_in_{P.dataset}_{ood_score}'
        save_path += f'_{P.t}'

        # Obtain the scores of IND data
        P.logger.print("**************IND**************")
        for _, id_loader in id_loaders.items():
            scores_id = get_scores(id_loader, score_func)

        if P.save_score:
            y = []
            for _, y_ in id_loader:
                y.append(y_)
            y = torch.cat(y).numpy()
            ys = np.sort(list(set(y)))
            for y_ in ys:
                idx = y == y_
                np.save(f'{save_path}_{y_}.npy', scores_id[idx])

        # For each OOD dataset, obtain scores, and compute AUC
        for ood, ood_loader in ood_loaders.items():
            P.logger.print("**************OUT**************")
            scores_ood = get_scores(ood_loader, score_func)
            auc = get_auroc(scores_id, scores_ood) * 100
            auroc_dict[ood][ood_score] = auc
            P.auc_tracker.update(auc,
                                int(P.logout.split('task_')[-1]),
                                int(ood.split('task_')[-1]))

            if P.save_score:
                np.save(f'{save_path}_out_{ood}.npy', scores_ood)

    return auroc_dict

def get_ood_score_func(P, model, ood_score, simclr_aug=None):
    def score_func(x, y):
        return compute_ood_score(P, model, ood_score, x, simclr_aug=simclr_aug, y=y)
    return score_func

def get_scores(loader, score_func):
    scores = []
    count, total, correct_task = 0, 0, 0
    for i, (x, y) in enumerate(loader):
        s = score_func(x.to(device), y=y)
        if isinstance(s, dict):
            try:
                correct_task += s['correct_task']
            except KeyError:
                correct_task += len(y) * -1

            count += s['count']
            s = s['scores']
            total += len(y)
        # assert s.dim() == 1 and s.size(0) == x.size(0)

        scores.append(s.detach().cpu().numpy())
    if total > 0:
        print("acc", count / total * 100)
    if correct_task > 0:
        print("task id prediction", correct_task / total * 100)
    return np.concatenate(scores)

def get_auroc(scores_id, scores_ood):
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    return roc_auc_score(labels, scores)

def compute_ood_score(P, model, ood_score, x, simclr_aug=None, y=None):
    model.eval()

    if ood_score == 'baseline_marginalized':
        total_outputs = 0
        for i in range(4):
            x_rot = torch.rot90(x, i, (2, 3))
            _, outputs_aux, masks = model(P.t, x_rot, s=P.smax, penultimate=True, joint=True)
            total_outputs += outputs_aux['joint'][:, P.n_cls_per_task * i:P.n_cls_per_task * (i + 1)]

        if P.cal_w is not None:
            total_outputs = total_outputs * P.cal_w[P.t] + P.cal_b[P.t]
            
        # score is based on MSP
        scores = F.softmax(total_outputs / 4., dim=1).max(dim=1)[0]
        return scores

    else:
        raise NotImplementedError()
