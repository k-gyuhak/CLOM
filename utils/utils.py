import os
import pickle
import random
import shutil
import sys
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value # THIS IS THE CURRENT VALUE
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count


def load_checkpoint(logdir, mode='last', loc=None):
    if mode == 'last':
        model_path = os.path.join(logdir, 'last.model')
        optim_path = os.path.join(logdir, 'last.optim')
        config_path = os.path.join(logdir, 'last.config')
    elif mode == 'best':
        model_path = os.path.join(logdir, 'best.model')
        optim_path = os.path.join(logdir, 'best.optim')
        config_path = os.path.join(logdir, 'best.config')

    else:
        raise NotImplementedError()

    print("=> Loading checkpoint from '{}'".format(logdir))
    if os.path.exists(model_path):
        model_state = torch.load(model_path, map_location=loc)
        optim_state = torch.load(optim_path, map_location=loc)
        with open(config_path, 'rb') as handle:
            cfg = pickle.load(handle)
    else:
        return None, None, None

    return model_state, optim_state, cfg


def save_checkpoint(epoch, model_state, optim_state, logdir):
    last_model = os.path.join(logdir, 'last.model')
    last_optim = os.path.join(logdir, 'last.optim')
    last_config = os.path.join(logdir, 'last.config')

    opt = {
        'epoch': epoch,
    }
    torch.save(model_state, last_model)
    torch.save(optim_state, last_optim)
    with open(last_config, 'wb') as handle:
        pickle.dump(opt, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_linear_checkpoint(logdir, mode='last'):
    if mode == 'last':
        linear_optim_path = os.path.join(logdir, 'last.linear_optim')
    elif mode == 'best':
        linear_optim_path = os.path.join(logdir, 'best.linear_optim')
    else:
        raise NotImplementedError()

    print("=> Loading linear optimizer checkpoint from '{}'".format(logdir))
    if os.path.exists(linear_optim_path):
        linear_optim_state = torch.load(linear_optim_path)
        return linear_optim_state
    else:
        return None


def save_linear_checkpoint(linear_optim_state, logdir):
    last_linear_optim = os.path.join(logdir, 'last.linear_optim')
    torch.save(linear_optim_state, last_linear_optim)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def normalize(x, dim=1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def make_model_diagrams(probs, labels, n_bins=10):
    """
    outputs - a torch tensor (size n x num_classes) with the outputs from the final linear layer
    - NOT the softmaxes
    labels - a torch tensor (size n) with the labels
    """
    confidences, predictions = probs.max(1)
    accuracies = torch.eq(predictions, labels)
    f, rel_ax = plt.subplots(1, 2, figsize=(4, 2.5))

    # Reliability diagram
    bins = torch.linspace(0, 1, n_bins + 1)
    bins[-1] = 1.0001
    width = bins[1] - bins[0]
    bin_indices = [confidences.ge(bin_lower) * confidences.lt(bin_upper) for bin_lower, bin_upper in
                   zip(bins[:-1], bins[1:])]
    bin_corrects = [torch.mean(accuracies[bin_index]) for bin_index in bin_indices]
    bin_scores = [torch.mean(confidences[bin_index]) for bin_index in bin_indices]

    confs = rel_ax.bar(bins[:-1], bin_corrects.numpy(), width=width)
    gaps = rel_ax.bar(bins[:-1], (bin_scores - bin_corrects).numpy(), bottom=bin_corrects.numpy(), color=[1, 0.7, 0.7],
                      alpha=0.5, width=width, hatch='//', edgecolor='r')
    rel_ax.plot([0, 1], [0, 1], '--', color='gray')
    rel_ax.legend([confs, gaps], ['Outputs', 'Gap'], loc='best', fontsize='small')

    # Clean up
    rel_ax.set_ylabel('Accuracy')
    rel_ax.set_xlabel('Confidence')
    f.tight_layout()
    return f


def md(data, mean, mat, inverse=False):
    if data.ndim == 1:
        data.reshape(1, -1)
    delta = (data - mean)

    if not inverse:
        mat = np.linalg.inv(mat)

    dist = np.dot(np.dot(delta, mat), delta.T)
    return np.sqrt(np.diagonal(dist)).reshape(-1, 1)

# class Printer:
#     def __init__(self, P):
#         self.printfn = P.printfn
#         self.P = P

#     def printf(self, *object, sep='', end='\n', flush=False):
#         print(*object, sep=sep, end=end, file=sys.stdout, flush=flush)

#         with open('./' + self.P.logout + '/' + self.printfn + '.txt', 'a') as f:
#             print(*object, sep=sep, end=end, file=f, flush=flush)

class Logger:
    def __init__(self, P):
        self.init = datetime.now()
        self.local_rank = P.local_rank
        self.P = P

        if P.load_path is None:
            if P.mode == 'sup_simclr_CSI' and P.t == 0:
                pass
            elif P.mode == 'sup_simclr_CSI' and P.t > 0:
                self.P.load_path = f'./logs/{P.dataset}/linear_task_{P.t - 1}'
            elif P.mode == 'sup_CSI_linear':
                self.P.load_path = f'./logs/{P.dataset}/feature_task_{P.t}'
            elif P.mode == 'test_marginalized_acc' or P.mode == 'ood':
                self.P.load_path = f'./logs/{P.dataset}/linear_task_{P.t}'
            elif P.mode == 'cil' or P.mode == 'cil_pre':
                self.P.load_path = f'./logs/{P.dataset}/linear_task_{P.cil_task}'
            else:
                raise NotImplementedError()

        if P.logout is None:
            if P.mode == 'sup_simclr_CSI' and P.t == 0:
                self.P.logout = f'./logs/{P.dataset}/feature_task_{P.t}'
            elif P.mode == 'sup_simclr_CSI' and P.t > 0:
                self.P.logout = f'./logs/{P.dataset}/feature_task_{P.t}'
            elif P.mode == 'sup_CSI_linear':
                self.P.logout = f'./logs/{P.dataset}/linear_task_{P.t}'
            elif P.mode == 'test_marginalized_acc' or P.mode == 'ood':
                self.P.logout = f'./logs/{P.dataset}/linear_task_{P.t}'
            elif P.mode == 'cil' or P.mode == 'cil_pre':
                self.P.logout = f'./logs/{P.dataset}/linear_task_{P.cil_task}'
            else:
                raise NotImplementedError()

        self._make_dir()

    def now(self):
        time = datetime.now()
        diff = time - self.init
        self.print(time.strftime("%m|%d|%Y %H|%M|%S"), f" | Total: {diff}")

    def print(self, *object, sep=' ', end='\n', flush=False, filename='/result.txt'):
        if self.local_rank == 0:
            print(*object, sep=sep, end=end, file=sys.stdout, flush=flush)

            if self.P.printfn is not None:
                filename = self.P.printfn
            with open(self.dir() + '/' + filename, 'a') as f:
                print(*object, sep=sep, end=end, file=f, flush=flush)

    def _make_dir(self):
        if self.local_rank == 0:
            if not os.path.isdir(self.dir()):
                os.makedirs(self.dir())

    def dir(self):
        return self.P.logout

    def time_interval(self):
        if self.local_rank == 0:
            self.print("Total time spent: {}".format(datetime.now() - self.init))

class Tracker:
    def __init__(self, P):
        self.print = P.logger.print
        self.mat = np.zeros((P.n_tasks * 2 + 1, P.n_tasks * 2 + 1)) - 100

    def update(self, acc, task_id, p_task_id):
        """
            acc: float, accuracy
            task_id: int, current task id
            p_task_id: int, previous task's task id
        """
        self.mat[task_id, p_task_id] = acc

        # Compute average
        self.mat[task_id, -1] = np.mean(self.mat[task_id, :p_task_id + 1])

        # Compute forgetting
        for i in range(task_id):
            self.mat[-1, i] = self.mat[i, i] - self.mat[task_id, i]

        # Compute average incremental accuracy
        self.mat[-1, -1] = np.mean(self.mat[:task_id + 1, -1])

    def print_result(self, task_id, type='acc', print=None):
        if print is None: print = self.print
        if type == 'acc':
            # Print accuracy
            for i in range(task_id + 1):
                for j in range(task_id + 1):
                    acc = self.mat[i, j]
                    if acc != -100:
                        print("{:.2f}\t".format(acc), end='')
                    else:
                        print("\t", end='')
                print("{:.2f}".format(self.mat[i, -1]))
        elif type == 'forget':
            # Print forgetting and average incremental accuracy
            for i in range(task_id + 1):
                acc = self.mat[-1, i]
                if acc != -100:
                    print("{:.2f}\t".format(acc), end='')
                else:
                    print("\t", end='')
            print("{:.2f}".format(self.mat[-1, -1]))
            if task_id > 0:
                forget = np.mean(self.mat[-1, :task_id])
                print("{:.2f}".format(forget))
        else:
            raise NotImplementedError("Type must be either 'acc' or 'forget'")

class AUCTracker:
    def __init__(self, P):
        self.print = P.logger.print
        self.mat = np.zeros((P.n_tasks * 2 + 1, P.n_tasks * 2 + 1)) - 100
        self.n_tasks = P.n_tasks
        self.last_id = 0

    def update(self, acc, task_id, p_task_id):
        """
            acc: float, accuracy
            task_id: int, current task id
            p_task_id: int, previous task's task id
        """
        self.last_id = max([self.last_id, p_task_id])

        self.mat[task_id, p_task_id] = acc

        # Compute average
        self.mat[task_id, -1] = np.mean(np.concatenate([
                                                        self.mat[task_id, :task_id],
                                                        self.mat[task_id, task_id + 1:self.last_id + 1]
                                                        ]))

        # # Compute forgetting
        # for i in range(task_id):
        #     self.mat[-1, i] = self.mat[i, i] - self.mat[task_id, i]

        # Compute average incremental accuracy
        self.mat[-1, -1] = np.mean(self.mat[:task_id + 1, -1])

    def print_result(self, task_id, type='acc', print=None):
        if print is None: print = self.print
        if type == 'acc':
            # Print accuracy
            for i in range(task_id + 1):
                for j in range(self.n_tasks):
                    acc = self.mat[i, j]
                    if acc != -100:
                        print("{:.2f}\t".format(acc), end='')
                    else:
                        print("\t", end='')
                print("{:.2f}".format(self.mat[i, -1]))
            # Print forgetting and average incremental accuracy
            for i in range(self.n_tasks):
                print("\t", end='')
            print("{:.2f}".format(self.mat[-1, -1]))
        else:
            raise NotImplementedError("Type must be 'acc'")

def compute_auc(in_scores, out_scores):
    from sklearn.metrics import roc_auc_score
    # Return auc e.g. auc=0.95
    if isinstance(in_scores, list):
        in_scores = np.concatenate(in_scores)
    if isinstance(out_scores, list):
        out_scores = np.concatenate(out_scores)

    labels = np.concatenate([np.ones_like(in_scores),
                             np.zeros_like(out_scores)])
    try:
        auc = roc_auc_score(labels, np.concatenate((in_scores, out_scores)))
    except ValueError:
        print("Input contains NaN, infinity or a value too large for dtype('float64').")
        auc = -0.99
    return auc

def auc(score_dict, task_id, auc_tracker):
    """
        AUC: AUC_ij = output values of task i's heads using i'th task data (IND)
                      vs output values of task i's head using j'th task data (OOD)
        NOTE 
    """
    in_scores = score_dict[task_id][:, task_id]

    for k, val in score_dict.items():
        if k != task_id:
            ood_scores = val[:, task_id]
            auc_value = compute_auc(in_scores, ood_scores)
            auc_tracker.update(auc_value * 100, task_id, k)