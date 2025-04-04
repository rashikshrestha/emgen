from dataclasses import dataclass
from emgen_config.train import TrainConfig
from emgen_config.dataset.dataset import ToyDatasetConfig, MNISTDatasetConfig


@dataclass
class NoiseSchedulerConfig:
    _target_: str = "emgen.generative_model.diffusion.noise_scheduler.NoiseScheduler"
    device: str = "${device}"
    num_timesteps: int = 100
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    diffusion_type: str = "ddpm"

#! Diffusion Model Architectures
@dataclass
class LinearArchConfig:
    _target_: str = "emgen.generative_model.diffusion.diffusion_model_arch.LinearArch"
    device: str = "${device}"
    data_dim: int = 2
    emb_size: int = 128
    input_emb: str = "sinusoidal"
    time_emb: str = "sinusoidal"
    hidden_size: int = 128
    hidden_layers: int = 3
    weights: str | None = None
    
@dataclass
class UNetArchConfig:
    _target_: str = "emgen.generative_model.diffusion.diffusion_model_arch.UNetArch"
    device: str = "${device}"
    emb_size: int = 128
    time_emb: str = "sinusoidal"
    in_channels: int = 1
    base_channels: int = 64
    num_down: int = 2
    weights: str | None = None
    
@dataclass
class UNet1DArchConfig:
    _target_: str = "emgen.generative_model.diffusion.diffusion_model_arch.UNet1DArch"


#! Diffusion Model  
@dataclass
class DiffusionModelConfig:
    _target_: str = "emgen.generative_model.diffusion.diffusion_model.DiffusionModel"
    device: str = "${device}"
    noise_scheduler: NoiseSchedulerConfig = NoiseSchedulerConfig()
    diffusion_arch: LinearArchConfig = LinearArchConfig() #TODO: add more type annotation here for unet archs
    dataset: ToyDatasetConfig = ToyDatasetConfig()
    train: TrainConfig = TrainConfig()