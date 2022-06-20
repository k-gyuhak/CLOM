import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cum_mask(P, t, model, p_mask):
    try:
        model = model.module
    except AttributeError:
        model = model

    task_id = torch.tensor([t]).to(device)
    mask = {}

    gc1, gc2, gc3, gc4 = model.mask(task_id, s=P.smax)

    mask['ec1'] = gc1.detach()
    mask['ec1'].requires_grad = False

    mask['ec2'] = gc2.detach()
    mask['ec2'].requires_grad = False

    mask['ec3'] = gc3.detach()
    mask['ec3'].requires_grad = False

    mask['ec4'] = gc4.detach()
    mask['ec4'].requires_grad = False

    if p_mask is None:
        p_mask = {}
        for n in mask.keys():
            p_mask[n] = mask[n]
    else:
        for n in mask.keys():
            p_mask[n] = torch.max(p_mask[n], mask[n])
    return p_mask

def freeze_mask(P, t, model, p_mask):
    try:
        model = model.module
    except AttributeError:
        model = model

    mask_back = {}

    for n, p in model.named_parameters():
        if n == 'conv1.weight':
            mask_back[n] = 1 - p_mask['ec1'].data.view(-1, 1, 1, 1).expand_as(p)
        elif n == 'conv1.bias':
            mask_back[n] = 1 - p_mask['ec1'].data.view(-1)
        elif n == 'conv2.weight':
            post = p_mask['ec2'].data.view(-1, 1, 1, 1).expand_as(p)
            pre  = p_mask['ec1'].data.view(1, -1, 1, 1).expand_as(p)
            mask_back[n] = 1 - torch.min(post, pre)
        elif n == 'conv2.bias':
            mask_back[n] = 1 - p_mask['ec2'].data.view(-1)
        elif n == 'conv3.weight':
            post = p_mask['ec3'].data.view(-1, 1, 1, 1).expand_as(p)
            pre  = p_mask['ec2'].data.view(1, -1, 1, 1).expand_as(p)
            mask_back[n] = 1 - torch.min(post, pre)
        elif n == 'conv3.bias':
            mask_back[n] = 1 - p_mask['ec3'].data.view(-1)
        elif n == 'fc1.weight':
            post = p_mask['ec4'].data.view(-1, 1).expand_as(p)
            pre  = p_mask['ec3'].data.view(-1, 1, 1).expand((p_mask['ec3'].size(1), 2, 2)).contiguous().view(1, -1).expand_as(p)
            mask_back[n] = 1 - torch.min(post, pre)
        elif n == 'fc1.bias':
            mask_back[n] = 1 - p_mask['ec4'].data.view(-1)
    return mask_back

    #     if n == 'fc1.weight':
    #         mask_back[n] = 1 - p_mask['ec1'].data.view(-1, 1).expand_as(p)
    #     elif n == 'fc1.bias':
    #         mask_back[n] = 1 - p_mask['ec1'].data.view(-1)
    #     elif n == 'fc2.weight':
    #         post = p_mask['ec2'].data.view(-1, 1).expand_as(p)
    #         pre  = p_mask['ec1'].data.view(1, -1).expand_as(p)
    #         mask_back[n] = 1 - torch.min(post, pre)
    #     elif n == 'fc2.bias':
    #         mask_back[n] = 1 - p_mask['ec2'].data.view(-1)
    #     elif n == 'fc3.weight':
    #         post = p_mask['ec3'].data.view(-1, 1).expand_as(p)
    #         pre  = p_mask['ec2'].data.view(1, -1).expand_as(p)
    #         mask_back[n] = 1 - torch.min(post, pre)
    #     elif n == 'fc3.bias':
    #         mask_back[n] = 1 - p_mask['ec3'].data.view(-1)
    # return mask_back

def cum_mask_simclr(P, t, model, p_mask):
    assert p_mask is not None

    try:
        model = model.module
    except AttributeError:
        model = model

    task_id = torch.tensor([t]).to(device)

    mask = {}
    for n, _ in model.named_parameters():
        if 'simclr_layer.ec' in n:
            n = '.'.join(n.split('.')[:-1])
            gc1 = model.simclr_layer.mask(task_id, s=P.smax)
            mask[n] = gc1.detach()
            mask[n].requires_grad = False

    if 'simclr_layer.ec' in p_mask.keys():
        for n in mask.keys():
            p_mask[n] = torch.max(p_mask[n], mask[n])
    else:
        for n in mask.keys():
            p_mask[n] = mask[n]
    return p_mask

def freeze_mask_simclr(P, t, model, p_mask, mask_back):
    assert mask_back is not None

    try:
        model = model.module
    except AttributeError:
        model = model

    for n, p in model.named_parameters():
        if n == 'simclr_layer.fc1.weight':
            post = p_mask['simclr_layer.ec'].data.view(-1, 1).expand_as(p)
            pre  = p_mask['ec4'].data.view(1, -1).expand_as(p)
            mask_back[n] = 1 - torch.min(post, pre)
        elif n == 'simclr_layer.fc1.bias':
            mask_back[n] = 1 - p_mask['simclr_layer.ec'].data.view(-1)
    return mask_back
