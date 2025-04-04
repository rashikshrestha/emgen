import torch
from torch import nn
import logging

from emgen.utils.positional_embeddings import PositionalEmbedding
from emgen.model.simple_linear import SimpleLinear
from emgen.model.unet import DownBlock, Bottleneck, UpBlock

log = logging.getLogger(__name__)


class LinearArch(nn.Module):
    """
    For given xt, it predicts x0 or the noise.
    """
    def __init__(
        self, 
        device: str = 'cpu',
        data_dim: int = 2,
        emb_size: int = 128,
        input_emb: str = "sinusoidal",
        time_emb: str = "sinusoidal",
        hidden_size: int = 128, 
        hidden_layers: int = 3,
        weights: str = None
    ):
        super().__init__()
        self.device = device

        #! Positional Embeddings
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0, device=self.device)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0, device=self.device)
        self.time_mlp   = PositionalEmbedding(emb_size, time_emb, device=self.device)

        #! Main Backbone
        input_size = len(self.input_mlp1.layer) + len(self.input_mlp2.layer) + len(self.time_mlp.layer)
        self.joint_mlp = SimpleLinear(input_size, hidden_size, hidden_layers, data_dim)
        
        if weights is not None:
            state_dict = torch.load(weights, map_location=self.device)
            self.load_state_dict(state_dict)
            log.info(f"Loaded weights from {weights} into LinearArch")


    def forward(self, x, t):
        """
        B = batch size, d = dimension
        
        Args:
            x (torch.Tensor): [B, d] Noisy sample
            t (torch.Tensor): [B,] Time step
        Returns:
            torch.Tensor: [b, d] Noise or x0 sample
        """
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
    def __init__(self,
        device: str = 'cpu',
        emb_size=128, 
        time_emb: str = "sinusoidal",
        in_channels=3, 
        base_channels=64, 
        num_down=2,
        weights: str = None
    ):
        """
        Args:
            in_channels (int): Number of input (and output) channels.
            base_channels (int): Number of channels at input and output of UNet
            num_down (int): Number of downsampling steps.
        """
        super().__init__()
        self.device = device
        
        #! Time embedding
        self.time_mlp = PositionalEmbedding(emb_size, time_emb, scale=1.0, device=self.device)
        
        #! Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        #! Encoder
        self.down_blocks = nn.ModuleList()
        channels = base_channels
        for _ in range(num_down):
            self.down_blocks.append(DownBlock(channels, channels * 2, emb_size))
            channels *= 2
        
        #! Bottleneck block
        self.bottleneck = Bottleneck(channels, channels, emb_size)
        
        #! Decoder
        self.up_blocks = nn.ModuleList()
        for _ in range(num_down):
            self.up_blocks.append(UpBlock(channels, channels // 2, emb_size))
            channels //= 2
        
        #! Final convolution
        self.final_conv = nn.Conv2d(channels, in_channels, kernel_size=1)
       
        #! Load weights if available 
        if weights is not None:
            state_dict = torch.load(weights, map_location=self.device)
            self.load_state_dict(state_dict)
            log.info(f"Loaded weights from {weights} into LinearArch")
   
    
    def forward(self, x, t):
        """
        B = batch size, C = image channels, H = height, W = width
        
        Args:
            x (torch.Tensor): [B, C, H, W] Noisy Image.
            t (torch.Tensor): [B,] Time step.
        Returns:
            torch.Tensor: [B, C, H, W] Noise or x0 sample.
        """
        #! Create time embedding for conditioning
        t_emb = self.time_mlp(t)  # t_emb shape: [B, emb_size]
        
        #! Initial convolution
        x = self.init_conv(x)
        
        #! Downsample
        skip_connections = []
        for down in self.down_blocks:
            skip, x = down(x, t_emb)
            skip_connections.append(skip)
        
        #! Bottleneck
        x = self.bottleneck(x, t_emb)
        
        #! Upsample
        for up in self.up_blocks:
            skip = skip_connections.pop()
            x = up(x, skip, t_emb)
        
        #! Final projection
        return self.final_conv(x)
     

class UNet1DArch(nn.Module):
    #TODO: Ghasideh
    def __init__(self):
        super().__init__()