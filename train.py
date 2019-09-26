import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.nn.utils import clip_grad_norm_


def _train_loop(
    model,
    dataloader,
    loss_criterion,
    optimiser,
    l2_clip_norm=None,
    lr_scheduler=None,
    use_gpu=False,
):
    outputs, losses = [], []
    for x, y in dataloader:
        if use_gpu:
            x, y = x.cuda(), y.cuda()

        out = model(x)
        loss = loss_criterion(out, y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if l2_clip_norm is not None:
            # Clip weights in final layer
            clip_grad_norm_(model.fc.parameters(), l2_clip_norm)

        outputs.append(out)
        losses.append(loss.item())

    if lr_scheduler is not None:
        lr_scheduler.step()

    return outputs, losses


def train_model(
    model,
    dataloader,
    n_epochs,
    l2_clip_norm=None,
    lr_decay=None,
    use_gpu=False,
):

    optimiser = optim.Adadelta(model.parameters())

    lr_scheduler = None
    if lr_decay is not None:
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimiser, gamma=lr_decay
        )
    loss = nn.CrossEntropyLoss()

    # Training Loop
    model.train()
    for e in range(n_epochs):
        _, batch_losses = _train_loop(
            model,
            dataloader,
            loss,
            optimiser,
            l2_clip_norm,
            lr_scheduler,
            use_gpu,
        )
        av_train_loss = np.average(batch_losses)
        print(f"Epoch {e + 1}: Training Loss {av_train_loss}")

    return model
