"""Genomics Benchmark CNN model.

Adapted from https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks/blob/main/src/genomic_benchmarks/models/torch.py
"""

import torch
from torch import nn


class GenomicsBenchmarkCNN(nn.Module):
    def __init__(self, number_of_classes, vocab_size, input_len, embedding_dim=100):
        """Genomics Benchmark CNN model.

        `embedding_dim` = 100 comes from:
        https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks/tree/main/experiments/torch_cnn_experiments
        """
        super(GenomicsBenchmarkCNN, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.cnn_model = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=16, kernel_size=8, bias=True),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=8, bias=True),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=8, bias=True),
            nn.BatchNorm1d(4),
            nn.MaxPool1d(2),

            nn.Flatten()
        )
        self.dense_model = nn.Sequential(
            nn.Linear(self.count_flatten_size(input_len), 512),
            # To be consistent with SSM classifier decoders, we use num_classes (even when it's binary)
            nn.Linear(512, number_of_classes)
        )

    def count_flatten_size(self, input_len):
        zeros = torch.zeros([1, input_len], dtype=torch.long)
        x = self.embeddings(zeros)
        x = x.transpose(1, 2)
        x = self.cnn_model(x)
        return x.size()[1]

    def forward(self, x, state=None):  # Adding `state` to be consistent with other models
        x = self.embeddings(x)
        x = x.transpose(1, 2)
        x = self.cnn_model(x)
        x = self.dense_model(x)
        return x, state  # Returning tuple to be consistent with other models
