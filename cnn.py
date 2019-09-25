"""
Contains code for CNN classifiers.
Move this code into ir-research.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from softmax import Softmax


class YKCNNClassifier(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        max_seq_length,
        output_dims=2,
        out_channels=100,
        embed_dim=300,
        padding_idx=0,
        kernel_heights=[3, 4, 5],
        hidden_dims=[],
        fc_dropout=0,
        embedding_matrix=None,
        freeze_embedding_layer=True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = 1
        self.n_kernels = len(kernel_heights)
        self.pool_sizes = [(max_seq_length - K, 1) for K in kernel_heights]
        self.max_seq_length = max_seq_length
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.fc_dropout = fc_dropout

        # Assumes vocab size is same as embedding matrix size. Therefore should
        # contain special tokens e.g. <pad>
        self.embedding = nn.Embedding(
            vocabulary_size, embed_dim, padding_idx=0
        )
        if embedding_matrix is not None:
            # Load pre-trained weights. Should be torch FloatTensor
            self.embedding.from_pretrained(embedding_matrix)

        if freeze_embedding_layer:
            self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=(K, embed_dim),
                )
                for K in kernel_heights
            ]
        )
        self.pools = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=pool_size)
                for pool_size in self.pool_sizes
            ]
        )
        self.fc = Softmax(
            input_dim=self.out_channels * self.n_kernels,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dims,
            dropout=self.fc_dropout,
        )

    def forward(self, x):
        """
        x: (batch_size, max_sequence_length)
        """
        batch_size = x.size(0)
        assert x.size(1) == self.max_seq_length

        # (batch, max_sequenece_length, embedding_dim)
        x = self.embedding(x)

        # adds input channel
        # (batch, 1, max_sequence_length, embedding_dim)
        x = x.unsqueeze(dim=1)

        out_tensors = []
        for i, (convs, pools) in enumerate(zip(self.convs, self.pools)):
            activation = pools(F.relu(convs(x)))
            out_tensors.append(activation)

        # Output from conv and pooling operation will be of size
        # (batch_size, out * n_kernels, 1, 1)
        x = torch.cat(out_tensors, dim=1)

        # Reshape to pass into fully connected
        x = x.view(batch_size, -1)
        return self.fc(x)

    def predict(self, x):
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=1)

    def predict_classes(self, x):
        predictions = self.predict(x)
        return torch.argmax(predictions, dim=1)
