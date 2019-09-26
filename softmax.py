from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class Softmax(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.hidden_layers = nn.ModuleList([])
        for h in hidden_dims:
            self.hidden_layers.append(nn.Linear(input_dim, h))
            input_dim = h

        self.out = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for h in self.hidden_layers:
            x = self.dropout(x)
            x = F.relu(h(x))

        return self.out(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=1)

    def predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        predictions = self.predict(x)
        return torch.argmax(predictions, dim=1)
