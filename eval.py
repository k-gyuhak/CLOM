import os
import sys
from common.eval import *

model.eval()

P.logger.print(P)

# Create/load Trackers for accuracies and OOD
P.cil_tracker = Tracker(P)
if os.path.exists(f'./logs/{P.dataset}/cil_acc_tracker'):
    P.cil_tracker.mat = torch.load(f'./logs/{P.dataset}/cil_acc_tracker')
P.cal_cil_tracker = Tracker(P)
if os.path.exists(f'./logs/{P.dataset}/cal_cil_acc_tracker'):
    P.cal_cil_tracker.mat = torch.load(f'./logs/{P.dataset}/cal_cil_acc_tracker')
P.til_tracker = Tracker(P)
if os.path.exists(f'./logs/{P.dataset}/til_acc_tracker'):
    P.til_tracker.mat = torch.load(f'./logs/{P.dataset}/til_acc_tracker')
P.auc_tracker = AUCTracker(P)
if os.path.exists(f'./logs/{P.dataset}/auc_tracker'):
    P.auc_tracker.mat = torch.load(f'./logs/{P.dataset}/auc_tracker')


# Train calibration
if P.mode == 'cil_pre':
    from evals import cil_pre
    assert P.cil_task is not None
    cil_pre(P, model, valid_loaders, 0, criterion, test_loaders=test_loaders)

# CIL inference
elif P.mode == 'cil':
    from evals import cil
    assert P.cil_task is not None

    if P.cal_w is not None:
        P.logger.print("Using calibration")
        P.logger.print("w:", P.cal_w.data)
        P.logger.print("b:", P.cal_b.data)
    else:
        P.logger.print("No calibration")

    with torch.no_grad():
        cil(P, model, test_loaders, 0, w=P.cal_w, b=P.cal_b)

# TIL of network P.t
elif P.mode == 'test_marginalized_acc':
    from evals import test_classifier
    with torch.no_grad():
        test_classifier(P, model, test_loaders, 0, marginal=True)

# AUC of network P.t (IND: dataset of task t, OOD: dataset of task j != t)
elif P.mode == 'ood':
    from evals import eval_ood_detection

    with torch.no_grad():
        auroc_dict = eval_ood_detection(P, model, test_loaders, ood_test_loader, P.ood_score,
                                        train_loader=None, simclr_aug=simclr_aug)

    mean_dict = dict()
    for ood_score in P.ood_score:
        mean = 0
        for ood in auroc_dict.keys():
            mean += auroc_dict[ood][ood_score]
        mean_dict[ood_score] = mean / len(auroc_dict.keys())
    auroc_dict['task_mean'] = mean_dict

    bests = []
    for ood in auroc_dict.keys():
        message = ''
        best_auroc = 0
        for ood_score, auroc in auroc_dict[ood].items():
            message += '[%s %s %.4f] ' % (ood, ood_score, auroc)
            if auroc > best_auroc:
                best_auroc = auroc
        message += '[%s %s %.4f] ' % (ood, 'best', best_auroc)
        if P.print_score:
            P.logger.print(message)
        bests.append(best_auroc)

    bests = map('{:.4f}'.format, bests)
    P.logger.print('\t'.join(bests))

else:
    raise NotImplementedError()

P.logger.print()
# if P.mode == 'ood':
#     P.logger.print("AUC result")
#     P.auc_tracker.print_result(P.t, type='acc')
if P.mode == 'cil':
    P.logger.print("CIL result")
    P.cil_tracker.print_result(P.cil_task, type='acc')
    P.cil_tracker.print_result(P.cil_task, type='forget')
if P.mode == 'cil_pre':
    P.logger.print("CIL result after calibration")
    P.cal_cil_tracker.print_result(P.cil_task, type='acc')
    P.cal_cil_tracker.print_result(P.cil_task, type='forget')
if P.mode == 'test_marginalized_acc':
    P.logger.print("TIL result")
    P.til_tracker.print_result(P.t, type='acc')
    P.til_tracker.print_result(P.t, type='forget')
P.logger.print()

torch.save(P.cil_tracker.mat, f'./logs/{P.dataset}/cil_acc_tracker')
torch.save(P.cal_cil_tracker.mat, f'./logs/{P.dataset}/cal_cil_acc_tracker')
torch.save(P.til_tracker.mat, f'./logs/{P.dataset}/til_acc_tracker')
torch.save(P.auc_tracker.mat, f'./logs/{P.dataset}/auc_tracker')

P.logger.print('\n\n\n\n\n\n\n\n')
