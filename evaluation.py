import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics as sk_metrics


def eval_model(
    model: nn.Module, x: torch.Tensor, use_gpu: bool = False
) -> torch.Tensor:

    model.eval()
    if use_gpu:
        x = x.cuda()

    return model.predict_classes(x)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return sk_metrics.accuracy_score(y_true, y_pred)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Returns positive and negative F1 score"""
    return sk_metrics.f1_score(y_true, y_pred)
