from dataclasses import dataclass

from generative_model.diffusion_model import DiffusionModelConfig

@dataclass
class EmGenConfig():
    generative_model: DiffusionModelConfig