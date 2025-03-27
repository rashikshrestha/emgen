import torch
from torch import nn


class Predictor_Linear(nn.Module):
    """
    For given xt, it predicts x0 or the noise.
    """
    def __init__(
        self, 
        data_dim: int = 2,
        emb_size: int = 128,
        input_emb: str = "sinusoidal",
        time_emb: str = "sinusoidal",
        hidden_size: int = 128, 
        hidden_layers: int = 3, 
    ):
        super().__init__()

        #! Positional Embeddings
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.time_mlp   = PositionalEmbedding(emb_size, time_emb)

        #! Main Backbome
        self.joint_mlp = nn.Sequential(*layers)


    def forward(self, x, t):
        #! Positional Embedding
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        
        #! Input for the backbone
        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        
        #! Backbone
        x = self.joint_mlp(x)
        
        return x