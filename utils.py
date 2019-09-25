import torch
from torch.utils.data import TensorDataset, DataLoader


def create_dataloader(x, y, batch_size=50, shuffle=True):
    """Creates pytorch dataloader"""
    dataset = TensorDataset(torch.LongTensor(x), torch.LongTensor(y))
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    return dataloader
