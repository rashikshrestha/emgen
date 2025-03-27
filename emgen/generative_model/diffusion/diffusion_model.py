import argparse
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np

#! emgen imports
from emgen.generative_model.diffusion.diffusion_model_arch import LinearArch
from emgen_config.generative_model.diffusion_model import DiffusionModelConfig


class DiffusionModel():
    def __init__(self, config: DiffusionModelConfig):
        pass
    
    
    def train(self):
        pass
    
    
    def sample(self):
        pass