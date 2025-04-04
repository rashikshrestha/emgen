import torch
import torch.nn as nn
import torch.nn.functional as F


class DownBlock(nn.Module):
    """
    Downsampling block that applies two convolutions,
    adds a time-embedding bias, and then downsamples.
    """
    def __init__(self, in_channels, out_channels, emb_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.time_mlp = nn.Linear(emb_size, out_channels)
    
    def  forward(self, x, t_emb):
        x = self.conv(x)
        t_emb = self.time_mlp(t_emb).view(x.shape[0], -1, 1, 1)
        x += t_emb
        x_downsampled = self.downsample(x)
        return x, x_downsampled


class Bottleneck(nn.Module):
    """
    Bottleneck block without spatial downsampling.
    """
    def __init__(self, in_channels, out_channels, emb_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
        self.time_mlp = nn.Linear(emb_size, out_channels)
    
    def forward(self, x, t_emb):
        h = self.conv(x)
        batch = h.shape[0]
        t_bias = self.time_mlp(t_emb).view(batch, -1, 1, 1)
        return h + t_bias


class UpBlock(nn.Module):
    """
    Upsampling block that uses a transposed convolution to upsample,
    concatenates with a skip connection, and then applies convolution layers.
    """
    def __init__(self, in_channels, out_channels, emb_size):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
        self.time_mlp = nn.Linear(emb_size, out_channels)
    
    def forward(self, x, skip, t_emb):
        x = self.upconv(x)
        # Concatenate along channel dimension (skip connection)
        x = torch.cat([x, skip], dim=1)
        h = self.conv(x)
        batch = h.shape[0]
        t_bias = self.time_mlp(t_emb).view(batch, -1, 1, 1)
        return h + t_bias