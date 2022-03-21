import pickle
import os

import numpy as np
import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from .cifar_mod import CIFAR10, CIFAR100
from .mnist_mod import MNIST

from utils.utils import set_random_seed


MNIST_SUPERCLASS = np.arange(10).reshape(5, 2).tolist()
CIFAR10_SUPERCLASS = np.arange(10).reshape(5, 2).tolist()
CIFAR100_SUPERCLASS_10T = np.arange(100).reshape(10, 10).tolist()
CIFAR100_SUPERCLASS_20T = np.arange(100).reshape(20, 5).tolist()
TINYIMAGENET_SUPERCLASS_5T = np.arange(200).reshape(5, 40).tolist()
TINYIMAGENET_SUPERCLASS_10T = np.arange(200).reshape(10, 20).tolist()

def get_transform(P):
    # Note: data augmentation is implemented in the layers
    # Hence, we only define the identity transformation here
    if 'tinyImagenet' in P.dataset:
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.ToTensor()

    return train_transform, test_transform

def get_dataset(P, dataset, test_only=False, image_size=None, download=False):
    train_transform, test_transform = get_transform(P=P)

    if dataset == 'mnist':
        image_size = (28, 28, 1)
        total_cls = 10
        n_cls_per_task = 2
        train_set = MNIST(P.data_path, train=True, download=download, transform=train_transform)
        test_set = MNIST(P.data_path, train=False, download=download, transform=test_transform)

    elif dataset == 'cifar10':
        image_size = (32, 32, 3)
        total_cls = 10
        n_cls_per_task = 2

        train_set = CIFAR10(P.data_path, train=True, download=download, transform=train_transform)
        test_set = CIFAR10(P.data_path, train=False, download=download, transform=test_transform)

    elif 'cifar100' in dataset:
        image_size = (32, 32, 3)
        total_cls = 100
        if P.dataset == 'cifar100_10t':
            n_cls_per_task = 10
        elif P.dataset == 'cifar100_20t':
            n_cls_per_task = 5
        train_set = CIFAR100(P.data_path, train=True, download=download, transform=train_transform)
        test_set = CIFAR100(P.data_path, train=False, download=download, transform=test_transform)

    elif 'tinyImagenet' in dataset:
        image_size = (32, 32, 3)
        total_cls = 200
        if P.dataset == 'tinyImagenet_5t':
            n_cls_per_task = 40
        elif P.dataset == 'tinyImagenet_10t':
            n_cls_per_task = 20
        train_dir = P.data_path + '/TinyImagenet/train'
        test_dir =  P.data_path + '/TinyImagenet/val_folders'
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        train_set.targets = np.array(train_set.targets)
        test_set.targets = np.array(test_set.targets)

    else:
        raise NotImplementedError()

    if test_only:
        return test_set
    else:
        return train_set, test_set, image_size, n_cls_per_task, total_cls

def get_superclass_list(dataset):
    if dataset == 'mnist':
        return MNIST_SUPERCLASS
    elif dataset == 'cifar10':
        return CIFAR10_SUPERCLASS
    elif dataset == 'cifar100_10t':
        return CIFAR100_SUPERCLASS_10T
    elif dataset == 'cifar100_20t':
        return CIFAR100_SUPERCLASS_20T
    elif dataset == 'tinyImagenet_5t':
        return TINYIMAGENET_SUPERCLASS_5T
    elif dataset == 'tinyImagenet_10t':
        return TINYIMAGENET_SUPERCLASS_10T
    else:
        raise NotImplementedError()


def get_subclass_dataset(P, dataset, classes, f_select=None, l_select=None, val=False, indices_dict=None):
    """
        f_select, l_select: float or int. If int, choose idx=sz - cal_size regardless of its value
    """

    # Calibration size is 10 per class for tiny-imagnet and 20 per class for other datasets
    if 'tinyImagenet' in P.dataset:
        cal_size = 10
    else:
        cal_size = 20

    if not isinstance(classes, list):
        classes = [classes]

    if indices_dict is None:
        indices_dict = {}
        for c in classes:
            indices_dict[c] = []

        for idx, tgt in enumerate(dataset.targets):
            if tgt in classes:
                indices_dict[tgt.item()].append(idx)
                # indices.append(idx)
    else:
        indices_dict_ = {}
        with open(indices_dict, 'rb') as file:
            indices_dict = pickle.load(file)
        for c in classes:
            indices_dict_[c] = indices_dict[c]
        indices_dict = indices_dict_
        del indices_dict_

    indices = []
    for k in indices_dict.keys(): # for each class, select the first f_select as training, and rest as test
        sz = len(indices_dict[k])
        if f_select is not None:
            if not isinstance(f_select, int):
                idx = int(sz * f_select)
            else:
                idx = sz - cal_size
            indices.append(indices_dict[k][:idx])
        elif l_select is not None:
            if not isinstance(l_select, int):
                idx = int(sz * l_select)
            else:
                idx = sz - cal_size
            if val:
                indices.append(indices_dict[k][idx:][:cal_size])
            else:
                indices.append(indices_dict[k][idx:])
        else:
            indices.append(indices_dict[k])

    indices = np.concatenate(indices)
    indices = indices.tolist()

    if f_select is not None and l_select is not None:
        raise KeyError("only one of f_select and l_select must be chosen")

    dataset = Subset(dataset, indices)
    return dataset
