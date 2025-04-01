import torch
from torch import nn
import logging

from emgen.utils.positional_embeddings import PositionalEmbedding
from emgen.model.simple_linear import SimpleLinear

log = logging.getLogger(__name__)

class LinearArch(nn.Module):
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
        weights: str = None
    ):
        super().__init__()

        #! Positional Embeddings
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.time_mlp   = PositionalEmbedding(emb_size, time_emb)

        #! Main Backbone
        input_size = len(self.input_mlp1.layer) + len(self.input_mlp2.layer) + len(self.time_mlp.layer)
        self.joint_mlp = SimpleLinear(input_size, hidden_size, hidden_layers, data_dim)
        
        if weights is not None:
            state_dict = torch.load(weights, map_location='cpu')
            self.load_state_dict(state_dict)
            log.info(f"Loaded weights from {weights} into LinearArch")
            input()


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
   
 
class UNetArch(nn.Module):
    #TODO: Ghasideh
    def __init__(self):
        super().__init__()


class UNet1DArch(nn.Module):
    #TODO: Ghasideh
    def __init__(self):
        super().__init__()