"""
Contains code for CNN classifiers.
Move this code into ir-research.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class YKCNNClassifier(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        max_seq_length,
        output_dims=2,
        in_channels=1,
        out_channels=100,
        embed_dim=300,
        padding_idx=0,
        kernel_heights=[3, 4, 5],
        dropout=0,
        embedding_matrix=None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.n_kernels = len(kernel_heights)
        self.pool_sizes = [(max_seq_length - K, 1) for K in kernel_heights]
        self.max_seq_length = max_seq_length
        self.output_dims = output_dims
        self.embedding = nn.Embedding(
            vocabulary_size, embed_dim, padding_idx=0
        )
        if embedding_matrix is not None:
            # Load pre-trained weights
            self.embedding.weight()

        self.dropout = nn.Dropout(dropout)
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
        self.fc = nn.Linear(self.out_channels * self.n_kernels, output_dims)

    def forward(self, x):
        """
        x: (batch_size, max_sequence_length)
        """
        batch_size = x.size(0)
        using_gpu = x.is_cuda
        assert x.size(1) == self.max_seq_length

        x = self.embedding(x)

        # adds input channel
        # x = (batch, n_input_channel, max_seq, embedding_dim)
        x = x.unsqueeze(dim=1)

        # Output from conv and pooling operation will be of size
        # (batch_size, out * n_kernels, 1, 1)
        concated_tensor = torch.zeros(
            (batch_size, self.out_channels * self.n_kernels, 1, 1)
        )
        if using_gpu:
            concated_tensor = concated_tensor.cuda()
        for i, (convs, pools) in enumerate(zip(self.convs, self.pools)):
            activation = pools(F.relu(convs(x)))
            concated_tensor[
                :, i * self.out_channels : (i + 1) * self.out_channels, :
            ] = activation

        # Reshape to pass into fully connected
        x = concated_tensor.view(concated_tensor.shape[0], -1)
        x = self.dropout(x)
        return self.fc(x)
