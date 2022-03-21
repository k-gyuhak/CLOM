import torch.nn as nn

from models.mlp import Net
from models.resnet import ResNet18
import models.transform_layers as TL


def get_simclr_augmentation(P, image_size):

    # parameter for resizecrop
    resize_scale = (P.resize_factor, 1.0) # resize scaling factor
    if P.resize_fix: # if resize_fix is True, use same scale
        resize_scale = (P.resize_factor, P.resize_factor)

    # Align augmentation
    color_jitter = TL.ColorJitterLayer(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8)
    color_gray = TL.RandomColorGrayLayer(p=0.2)
    resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=image_size)

    transform = nn.Sequential(
        color_jitter,
        color_gray,
        resize_crop,
    )

    return transform


def get_shift_module(P, eval=False):

    if P.shift_trans_type == 'rotation':
        shift_transform = TL.Rotation()
        K_shift = 4
    else:
        shift_transform = nn.Identity()
        K_shift = 1

    if not eval and not ('sup' in P.mode):
        assert P.batch_size == int(128/K_shift)

    return shift_transform, K_shift # shift_transform is a class


def get_shift_classifer(n_tasks, model, K_shift):
    model.shift_cls_layer = nn.ModuleList()
    for _ in range(n_tasks):
        model.shift_cls_layer.append(nn.Linear(model.last_dim, K_shift))

    return model


def get_classifier(P, mode, n_classes=10):
    if mode == 'resnet18':
        classifier = ResNet18(P, num_classes=n_classes)
    elif mode == 'mlp':
        classifier = Net(P, num_classes=n_classes)
    else:
        raise NotImplementedError()

    return classifier

