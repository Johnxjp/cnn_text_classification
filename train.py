import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def _train_step(
    model,
    dataloader,
    loss_criterion,
    optimiser,
    lr_scheduler,
    l2_norm_clip=0,
    use_gpu=False,
):
    """One epoch of training"""
    outputs, losses = [], []
    for x, y in dataloader:
        if use_gpu:
            x, y = x.cuda(), y.cuda()

        out = model(x)
        loss = loss_criterion(out, y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if l2_norm_clip > 0:
            # Clip weights in final layer
            with torch.no_grad():
                norms = torch.norm(model.fc.out.weight, dim=1, keepdim=True)
                norms_clipped = torch.clamp_max(norms, l2_norm_clip)

                # Renormalise weights
                model.fc.out.weight.div_(norms).mul_(norms_clipped)

        outputs.append(out)
        losses.append(loss.item())

    lr_scheduler.step()
    return outputs, losses


def train_model(
    model, dataloader, n_epochs, l2_norm_clip=0, lr_decay=1, use_gpu=False
):
    optimiser = optim.Adadelta(model.parameters())
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=lr_decay)
    loss = nn.CrossEntropyLoss()

    # Training Loop
    model.train()
    for e in range(n_epochs):
        _, batch_losses = _train_step(
            model,
            dataloader,
            loss,
            optimiser,
            lr_scheduler,
            l2_norm_clip,
            use_gpu,
        )
        av_train_loss = np.average(batch_losses)
        print(f"Epoch {e + 1}: Training Loss = {av_train_loss:.4}")

    return model
