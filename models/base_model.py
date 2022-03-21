from abc import *
import torch
import torch.nn as nn

class SimclrLayer(nn.Module):
    """ contrastivie projection layers """
    def __init__(self, P, last_dim, simclr_dim):
        super(SimclrLayer, self).__init__()

        self.fc1 = nn.Linear(last_dim, last_dim)
        self.relu = nn.ReLU()
        self.last = nn.ModuleList()

        # Initialize embeddings/heads for all the future tasks for convenience.
        # However, it works fine without knowing the number of tasks.
        # Just append a new module when necessary.
        for _ in range(P.n_tasks):
            self.last.append(nn.Linear(last_dim, simclr_dim))

        # Protect the projection parameters by using hard attention.
        # However, not sure if it's necessary as we don't really use it for inference.
        # Protecting/using previously learned parameters possibly encourages faster convergence.
        # TODO: Try without protection.
        self.ec = nn.ParameterList()
        for _ in range(P.n_tasks):
            self.ec.append(nn.Parameter(torch.randn(1, last_dim)))
        self.gate = torch.sigmoid

    def mask(self, t, s=1):
        gc1 = self.gate(s * self.ec[t])
        return gc1

    def mask_out(self, out, mask):
        out = out * mask.expand_as(out)
        return out

    def forward(self, t, features, s=1):
        gc1 = self.mask(t, s=s)

        out = self.fc1(features)
        out = self.relu(out)
        out = self.mask_out(out, gc1)
        out = self.last[t](out)
        return out, gc1

class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, last_dim, P, num_classes=10, simclr_dim=128):
        super(BaseModel, self).__init__()
        self.P = P

        # Standard classifier without ensemble. This is only for reference during feature training.
        # Not necessary for testing. Does not affect the final result with/without it during training.
        self.linear = nn.ModuleList()
        for _ in range(P.n_tasks):
            self.linear.append(nn.Linear(last_dim, num_classes))

        # Contrastive projection
        self.simclr_layer = SimclrLayer(P, last_dim, simclr_dim)

        # Independent ensemble classifier
        self.joint_distribution_layer = nn.ModuleList()
        for _ in range(P.n_tasks):
            self.joint_distribution_layer.append(nn.Linear(last_dim, 4 * num_classes))

    @abstractmethod
    def penultimate(self, t, inputs, s, all_features=False):
        pass

    def forward(self, t, inputs, s=1, penultimate=False, simclr=False, shift=False, joint=False):
        _aux = {}
        _return_aux = False

        features, masks = self.penultimate(t, inputs, s)

        output = self.linear[t](features)

        if penultimate:
            _return_aux = True
            _aux['penultimate'] = features

        if simclr:
            _return_aux = True
            out, gc1 = self.simclr_layer(t, features, s)
            masks.append(gc1)
            _aux['simclr'] = out

        if joint:
            _return_aux = True
            _aux['joint'] = self.joint_distribution_layer[t](features)

        if _return_aux:
            return output, _aux, masks

        return output, masks
