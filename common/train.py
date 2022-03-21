import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from .sgd_hat import SGD_hat
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from common.common import parse_args
import models.classifier as C
from datasets import get_dataset, get_superclass_list, get_subclass_dataset
from utils.utils import load_checkpoint, Logger

P = parse_args()

P.logger = Logger(P)

if torch.cuda.is_available():
    torch.cuda.set_device(P.local_rank)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

# Check if multi-gpu or not
P.n_gpus = torch.cuda.device_count()
P.logger.print("PNgpu", P.n_gpus)
if P.n_gpus > 1:
    import apex
    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler

    P.multi_gpu = True
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=P.n_gpus,
        rank=P.local_rank,
    )
else:
    P.multi_gpu = False

### only use one ood_layer while training
P.ood_layer = P.ood_layer[0]

### Initialize dataset ###
train_set_, _, image_size, n_cls_per_task, total_cls = get_dataset(P, dataset=P.dataset, download=True)
P.image_size = image_size
P.n_cls_per_task = n_cls_per_task
P.total_cls = total_cls
P.n_tasks = int(total_cls // n_cls_per_task)

cls_list = get_superclass_list(P.dataset)

train_set = get_subclass_dataset(P, train_set_, classes=cls_list[P.t], f_select=0.9, indices_dict=P.indices_dict)
test_set = get_subclass_dataset(P, train_set_, classes=cls_list[P.t], l_select=0.9, indices_dict=P.indices_dict)

kwargs = {'pin_memory': False, 'num_workers': 15}

if P.multi_gpu:
    train_sampler = DistributedSampler(train_set, num_replicas=P.n_gpus, rank=P.local_rank)
    test_sampler = DistributedSampler(test_set, num_replicas=P.n_gpus, rank=P.local_rank)
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=P.batch_size, **kwargs)
    test_loaders = {P.t: DataLoader(test_set, sampler=test_sampler, batch_size=P.test_batch_size, **kwargs)}
else:
    train_loader = DataLoader(train_set, shuffle=True, batch_size=P.batch_size, **kwargs)
    test_loaders = {P.t: DataLoader(test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)}

### Initialize model ###
simclr_aug = C.get_simclr_augmentation(P, image_size=P.image_size).to(device)
P.shift_trans, P.K_shift = C.get_shift_module(P, eval=True)
P.shift_trans = P.shift_trans.to(device)

model = C.get_classifier(P, P.model, n_classes=P.n_cls_per_task).to(device)
model = C.get_shift_classifer(P.n_tasks, model, P.K_shift).to(device)

criterion = nn.CrossEntropyLoss().to(device)

if P.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay)
    lr_decay_gamma = 0.1
elif P.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=P.lr_init, betas=(.9, .999), weight_decay=P.weight_decay)
    lr_decay_gamma = 0.3
elif P.optimizer == 'lars':
    from torchlars import LARS
    base_optimizer = SGD_hat(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay)
    optimizer = LARS(base_optimizer, eps=1e-8, trust_coef=0.001)
    lr_decay_gamma = 0.1
else:
    raise NotImplementedError()

if P.lr_scheduler == 'cosine':
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, P.epochs)
elif P.lr_scheduler == 'step_decay':
    milestones = [int(0.5 * P.epochs), int(0.75 * P.epochs)]
    scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=lr_decay_gamma, milestones=milestones)
else:
    raise NotImplementedError()

from training.scheduler import GradualWarmupScheduler
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=10.0, total_epoch=P.warmup, after_scheduler=scheduler)

# if resume_path is for resuming training
if P.resume_path is not None:
    resume = True
    model_state, optim_state, config = load_checkpoint(P.resume_path, mode='last')
    model.load_state_dict(model_state, strict=not P.no_strict)
    optimizer.load_state_dict(optim_state)
    start_epoch = config['epoch']
    best = 100
    error = 100.0
else:
    resume = False
    start_epoch = 1
    best = 100.0
    error = 100.0

# If training feature extractor of task > 0 or linear classifier (after training feature extractor),
# load model trained in previous step
if P.mode == 'sup_CSI_linear' or P.t > 0:
    assert P.load_path is not None

    loc = None
    if P.multi_gpu:
        if torch.cuda.is_available():
            loc = {'cuda:0': 'cuda',
                   'cuda:1': 'cuda',
                   'cuda:2': 'cuda',
                   'cuda:3': 'cuda'}
        else:
            loc = {'cuda:0': 'cpu',
                   'cuda:1': 'cpu',
                   'cuda:2': 'cpu',
                   'cuda:3': 'cpu'}

    checkpoint = torch.load(os.path.join(P.load_path, 'last.model'), map_location=loc)
    model.load_state_dict(checkpoint, strict=not P.no_strict)

if P.multi_gpu:
    simclr_aug = apex.parallel.DistributedDataParallel(simclr_aug, delay_allreduce=True)
    model = apex.parallel.convert_syncbn_model(model)
    model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)

# Load masks.
if P.load_path is None:
    p_mask = None
    mask_back = None
else:
    P.logger.print("Loading previous masks")

    if P.multi_gpu:
        if torch.cuda.is_available():
            loc = {'cuda:0': 'cuda',
                   'cuda:1': 'cuda',
                   'cuda:2': 'cuda'}
        else:
            loc = {'cuda:0': 'cpu',
                   'cuda:1': 'cpu',
                   'cuda:2': 'cpu'}

    mask_checkpoints = torch.load(os.path.join(P.load_path, 'masks'), map_location=loc)
    p_mask = mask_checkpoints['p_mask']
    mask_back = mask_checkpoints['mask_back']

    if P.multi_gpu:
        for n, p in model.module.named_parameters():
            p.grad = None
            if n in mask_back.keys():
                p.hat = mask_back[n]
            else:
                p.hat = None
    else:
        for n, p in model.named_parameters():
            p.grad = None
            if n in mask_back.keys():
                p.hat = mask_back[n]
            else:
                p.hat = None
