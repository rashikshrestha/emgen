from dataclasses import dataclass
from emgen_config.train import TrainConfig
from emgen_config.dataset.dataset import ToyDatasetConfig

@dataclass
class NoiseSchedulerConfig:
    _target_: str = "emgen.generative_model.diffusion.noise_scheduler.NoiseScheduler"
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    diffusion_type: str = "ddpm"

#! Diffusion Model Architecture
@dataclass
class LinearArchConfig:
    _target_: str = "emgen.generative_model.diffusion.diffusion_model_arch.LinearArch"
    data_dim: int = 2
    emb_size: int = 128
    input_emb: str = "sinusoidal"
    time_emb: str = "sinusoidal"
    hidden_size: int = 128
    hidden_layers: int = 3
    
    
@dataclass
class UNetArchConfig:
    _target_: str = "emgen.generative_model.diffusion.diffusion_model_arch.UNetArch"
    
    
@dataclass
class UNet1DArchConfig:
    _target_: str = "emgen.generative_model.diffusion.diffusion_model_arch.UNet1DArch"


#! Diffusion Model  
@dataclass
class DiffusionModelConfig:
    _target_: str = "emgen.generative_model.diffusion.diffusion_model.DiffusionModel"
    noise_scheduler: NoiseSchedulerConfig = NoiseSchedulerConfig()
    diffusion_arch: LinearArchConfig = LinearArchConfig() #TODO: add more type annotation here for unet archs
    dataset: ToyDatasetConfig = ToyDatasetConfig()
    train: TrainConfig = TrainConfig()