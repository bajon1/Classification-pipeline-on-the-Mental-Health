import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout_p, activation):
        super().__init__()
        self.input = nn.Linear(input_size, hidden_layers[0])

        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_layers[i+1], hidden_layers[i]) for i in range(len(hidden_layers))
        ])

        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=dropout_p)
        self.activation = nn.ReLU if activation == "relu" else nn.Tanh

    def forward(self, x):
        x = self.activation(self.input(x))
        for layer in self.hidden_layers:
            x = self.dropout(self.activation(layer(x)))

        return self.output(x)