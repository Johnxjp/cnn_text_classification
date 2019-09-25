import torch
from torch.utils.data import TensorDataset, DataLoader


def create_dataloader(x, y=None, batch_size=50, shuffle=True):
    """Creates pytorch dataloader"""
    if y is None:
        dataset = TensorDataset(torch.LongTensor(x))
    else:
        dataset = TensorDataset(torch.LongTensor(x), torch.LongTensor(y))
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    return dataloader
