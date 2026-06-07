import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout_p, activation):
        super().__init__()

    def forward(self, x):
        return x