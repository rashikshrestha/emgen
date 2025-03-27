import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x)) # Residual connection


class SimpleLinear(nn.Module):
    def __init__(
        self, 
        input_size: int,
        hidden_size: int, 
        hidden_layers: int, 
        output_size: int
    ):
        super().__init__()

        #! Input Layer
        layers = [nn.Linear(input_size, hidden_size), nn.GELU()]
       
        #! Hidden Layers (with Residual Connections) 
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
           
        #! Output Layer 
        layers.append(nn.Linear(hidden_size, output_size))
       
        #! Assemble all layers 
        self.joint_mlp = nn.Sequential(*layers)


    def forward(self, x):
        return self.joint_mlp(x)