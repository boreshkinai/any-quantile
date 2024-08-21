import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(torch.nn.Module):
    """Fully connected residual block"""

    def __init__(self, num_layers: int, layer_width: int, size_in: int, size_out: int, activation = F.relu):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.size_in = size_in
        self.size_out = size_out
        self.activation = activation
        
        assert num_layers > 1, f"MLP has to have at least 2 layers, {num_layers} specified"

        self.fc_layers = [torch.nn.Linear(size_in, layer_width)]
        self.fc_layers += [torch.nn.Linear(layer_width, layer_width) for _ in range(num_layers - 2)]
        self.fc_layers += [torch.nn.Linear(layer_width, size_out)]
        self.fc_layers = torch.nn.ModuleList(self.fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.fc_layers[:-1]:
            h = self.activation(layer(h))
        return self.fc_layers[-1](h)