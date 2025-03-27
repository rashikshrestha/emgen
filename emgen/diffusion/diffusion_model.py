import argparse
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

#! emgen imports
from emgen.diffusion.diffusion_model_arch import LinearArch

@dataclass
class DiffusionModelConfig:
    experiment_name: str = "base"
    timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    model_arch: str = "linear"
    diffusion_type: str = "ddpm"


class DiffusionModel():
    def __init__(self, config: DiffusionModelConfig):
        pass
    
    
    def train(self):
        pass
    
    
    def sample(self):
        pass