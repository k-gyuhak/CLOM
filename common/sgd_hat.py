import torch
from torch.optim.optimizer import required
from torch.optim import SGD

class SGD_hat(SGD):
    def __init__(self, params, lr=required, momentum=0, dampening=0, 
                 weight_decay=0, nesterov=False):
        super(SGD_hat, self).__init__(params, lr, momentum, dampening,
                                      weight_decay, nesterov)

    @torch.no_grad()
    def step(self, closure=None, hat=False):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                # temp = torch.ones(d_p.size()).to(d_p.device)
                # if len(d_p.size()) > 1:
                #     # temp = torch.sum(d_p, dim=1).detach()
                #     temp = d_p.detach()
                #     temp = torch.tensor(temp > 1e-30, dtype=torch.int) + torch.tensor(temp < -1e-30, dtype=torch.int)
                #     temp = temp.to(d_p.device)
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                if hat:
                    if p.hat is not None:
                        d_p = d_p * p.hat
                # print(d_p.size(), temp.size())
                # print(temp.expand_as(d_p), temp.expand_as(d_p).size())
                # d_p = d_p * temp#.expand_as(d_p)

                p.add_(d_p, alpha=-group['lr'])

        return loss