import os
import sys
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.common import parse_args
import models.classifier as C
from datasets import get_dataset, get_superclass_list, get_subclass_dataset
from utils.utils import Logger, Tracker, AUCTracker

P = parse_args()

P.logger = Logger(P)

P.n_gpus = torch.cuda.device_count()
assert P.n_gpus <= 1  # no multi GPU
P.multi_gpu = False

if torch.cuda.is_available():
    torch.cuda.set_device(P.local_rank)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

train_set, test_set, image_size, n_cls_per_task, total_cls = get_dataset(P, dataset=P.dataset, download=True)
cls_list = get_superclass_list(P.dataset)
P.n_superclasses = len(cls_list)

kwargs = {'pin_memory': False, 'num_workers': 15}
P.image_size = image_size
P.n_cls_per_task = n_cls_per_task
P.total_cls = total_cls
P.n_tasks = int(total_cls // n_cls_per_task)


if P.mode == 'cil' or P.mode == 'test_marginalized_acc':
    test_loaders = {}
    if P.all_dataset:
        # report the accuracies of all the tasks learned so far
        last_learned_task = int(P.load_path.split('task_')[-1])

        for p_task_id in range(last_learned_task + 1):
            # Obtain test data
            if P.validation:
                P.logger.print("Evaluation on validation set")
                test_subset = get_subclass_dataset(P, train_set, classes=cls_list[p_task_id], l_select=0.9)
            else:
                test_subset = get_subclass_dataset(P, test_set, classes=cls_list[p_task_id])
            test_loaders[p_task_id] = DataLoader(test_subset, shuffle=False, batch_size=P.test_batch_size, **kwargs)
    else:
        # report the accuracy of task P.t
        # Obtain test data
        if P.validation:
            P.logger.print("Evaluation on validation set")
            test_subset = get_subclass_dataset(P, train_set, classes=cls_list[P.t], l_select=0.9)
        else:
            test_subset = get_subclass_dataset(P, test_set, classes=cls_list[P.t])
        test_loaders[P.t] = DataLoader(test_subset, shuffle=False, batch_size=P.test_batch_size, **kwargs)

# training for output calibration
elif P.mode == 'cil_pre':
    # train_set, test_set, image_size, n_cls_per_task, total_cls = get_dataset(P, dataset=P.dataset)

    if P.dataset == 'mnist':
        # Provide some random integer for mnist for calibration.
        # If integer is provided for f_select or l_select, calibration size is determined as n_samples-20 for mnist
        # FIX ME: not working
        l_select = 5980
    else:
        l_select = 0.9

    valid_loaders, test_loaders = {}, {}
    for t in range(P.cil_task + 1):
        # This dataset is used for training calibration params. Each class has 20 samples for MNIST, CIFAR and 10 samples for T-Imagenet
        valid_subset = get_subclass_dataset(P, train_set, classes=cls_list[t], l_select=l_select, val=True, indices_dict=P.indices_dict) # if val=True, select only the last 20 samples for each class
        valid_loaders[t] = DataLoader(valid_subset, shuffle=True, batch_size=15, **kwargs)

        test_subset = get_subclass_dataset(P, test_set, classes=cls_list[t])
        test_loaders[t] = DataLoader(test_subset, shuffle=True, batch_size=200, **kwargs)

elif P.mode == 'ood':
    # Remove IND task from list of tasks
    P.ood_dataset = list(range(P.n_superclasses))
    P.ood_dataset.remove(P.t)

    test_loaders = {}
    if P.validation:
        P.logger.print("Evaluation on validation set")
        test_subset = get_subclass_dataset(P, train_set, classes=cls_list[P.t], l_select=0.9)
    else:
        test_subset = get_subclass_dataset(P, test_set, classes=cls_list[P.t])
    test_loaders[P.t] = DataLoader(test_subset, shuffle=False, batch_size=P.test_batch_size, **kwargs)

    # Save OOD dataloaders in a dictionary
    ood_test_loader = dict()
    for ood in P.ood_dataset:
        if P.validation:
            ood_test_subset = get_subclass_dataset(P, test_set, classes=cls_list[ood], l_select=0.9)
        else:
            ood_test_subset = get_subclass_dataset(P, test_set, classes=cls_list[ood])

        ood = f'task_{ood}'
        ood_test_loader[ood] = DataLoader(ood_test_subset, shuffle=False, batch_size=P.test_batch_size, **kwargs)


simclr_aug = C.get_simclr_augmentation(P, image_size=P.image_size).to(device)
P.shift_trans, P.K_shift = C.get_shift_module(P, eval=True)
P.shift_trans = P.shift_trans.to(device)

model = C.get_classifier(P, P.model, n_classes=P.n_cls_per_task).to(device)
model = C.get_shift_classifer(P.n_tasks, model, P.K_shift).to(device)
criterion = nn.CrossEntropyLoss().to(device)

if torch.cuda.is_available():
    loc = {'cuda:0': 'cuda',
           'cuda:1': 'cuda',
           'cuda:2': 'cuda'}
else:
    loc = {'cuda:0': 'cpu',
           'cuda:1': 'cpu',
           'cuda:2': 'cpu'}

# Load model
checkpoint = torch.load(os.path.join(P.load_path, 'last.model'), map_location=loc)
model.load_state_dict(checkpoint, strict=not P.no_strict)

# # Load masks (not necessary for inference)
# mask_checkpoints = torch.load(os.path.join(P.load_path, 'masks'), map_location=loc)
# p_mask = mask_checkpoints['p_mask']
# mask_back = mask_checkpoints['mask_back']

# If mode is cil and the calibration params are already trained, load them
try:
    if not P.disable_cal:
        trained_cal = torch.load(os.path.join(P.load_path, 'calibration'), map_location=loc)
        P.cal_w = trained_cal['w']
        P.cal_b = trained_cal['b']
except FileNotFoundError:
    pass
