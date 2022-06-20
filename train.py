import sys

from utils.utils import Logger
from utils.utils import save_checkpoint
from utils.utils import save_linear_checkpoint

from common.train import *
from evals import test_classifier

P.logger.print(P)

if P.model == 'resnet18':
    from mask_ops import *
elif P.model == 'mlp':
    from mask_ops_mlp import *
else:
    raise NotImpelementedError("HAT is only impelemented for ResNet-18")
# from mask_ops import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if P.mode == 'sup_CSI_linear':
    from training.sup.sup_CSI_linear import train
elif P.mode == 'sup_simclr_CSI':
    from training.sup.sup_simclr_CSI import train
else:
    raise NotImplementedError()

if P.multi_gpu:
    linear = model.module.linear
else:
    linear = model.linear

linear_optim = torch.optim.Adam(linear.parameters(), lr=1e-3, betas=(.9, .999), weight_decay=P.weight_decay)

# Training starts
try:
    for epoch in range(start_epoch, P.epochs + 1):
        P.logger.print(f'Epoch {epoch}', P.logout)
        model.train()

        if P.multi_gpu:
            train_sampler.set_epoch(epoch)

        kwargs = {}
        kwargs['linear'] = linear
        kwargs['linear_optim'] = linear_optim
        kwargs['simclr_aug'] = simclr_aug

        train(P, epoch, model, criterion, optimizer, scheduler_warmup, train_loader, p_mask, mask_back, **kwargs)

        model.eval()

        if epoch % P.save_step == 0 and P.local_rank == 0:
            if P.multi_gpu:
                save_states = model.module.state_dict()
            else:
                save_states = model.state_dict()

            save_checkpoint(epoch, save_states, optimizer.state_dict(), P.logout)
            save_linear_checkpoint(linear_optim.state_dict(), P.logout)

        if epoch % P.error_step == 0 and ('sup' in P.mode):
            error = test_classifier(P, model, test_loaders, epoch)

            is_best = (best > error)
            if is_best:
                best = error

            P.logger.print('[Epoch %3d] [Test %5.2f] [Best %5.2f]' % (epoch, error, best))

except KeyboardInterrupt:
    P.logger.print()

# Update and save masks
if P.local_rank == 0:
    checkpoint = torch.load(P.logout + '/last.model', map_location=None)

    if P.multi_gpu:
        model.module.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    p_mask = cum_mask(P, P.t, model, p_mask)
    mask_back = freeze_mask(P, P.t, model, p_mask)
    p_mask = cum_mask_simclr(P, P.t, model, p_mask)
    mask_back = freeze_mask_simclr(P, P.t, model, p_mask, mask_back)
    checkponts = {'p_mask': p_mask,
                  'mask_back': mask_back}
    torch.save(checkponts, P.logout + '/masks')
    P.logger.print("Saved masks")
P.logger.print('\n\n\n\n\n\n\n\n')
