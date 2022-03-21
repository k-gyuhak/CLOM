'''ResNet in PyTorch.
BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel
from models.transform_layers import NormalizeLayer
from torch.nn.utils import spectral_norm

class Net(BaseModel):
    # 784 -> 800 -> 800 -> last i.e. 4 layers
    def __init__(self, P, num_classes):
        last_dim = 800
        super(Net, self).__init__(last_dim, P, num_classes)
        self.last_dim = 800

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3) 
        self.ec1 = nn.ParameterList()
        for _ in range(P.n_tasks):
            self.ec1.append(nn.Parameter(torch.randn(1, 64)))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.ec2 = nn.ParameterList()
        for _ in range(P.n_tasks):
            self.ec2.append(nn.Parameter(torch.randn(1, 128)))

        self.conv3 = nn.Conv2d(128, 256, kernel_size=2)
        self.ec3 = nn.ParameterList()
        for _ in range(P.n_tasks):
            self.ec3.append(nn.Parameter(torch.randn(1, 256)))

        self.fc1 = nn.Linear(1024, 800)
        self.ec4 = nn.ParameterList()
        for _ in range(P.n_tasks):
            self.ec4.append(nn.Parameter(torch.randn(1, 800)))

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

        self.gate = torch.sigmoid

        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.5)

    def mask(self, t, s=1):
        gc1 = self.gate(s * self.ec1[t])
        gc2 = self.gate(s * self.ec2[t])
        gc3 = self.gate(s * self.ec3[t])
        gc4 = self.gate(s * self.ec4[t])
        return [gc1, gc2, gc3, gc4]

    def mask_out(self, out, mask):
        if len(out.size()) == 4:
            out = out * mask.view(1, -1, 1, 1).expand_as(out)
        else:
            out = out * mask.expand_as(out)
        return out

    def penultimate(self, t, x, s=1):
        masks = self.mask(t, s=s)
        gc1, gc2, gc3, gc4 = masks

        out = self.maxpool(self.drop1(self.relu(self.conv1(x))))
        out = self.mask_out(out, gc1)

        out = self.maxpool(self.drop1(self.relu(self.conv2(out))))
        out = self.mask_out(out, gc2)

        out = self.maxpool(self.drop2(self.relu(self.conv3(out))))
        out = self.mask_out(out, gc3)

        out = out.view(out.size(0), -1)

        out = self.drop2(self.relu(self.fc1(out)))
        out = self.mask_out(out, gc4)

        return out, masks


