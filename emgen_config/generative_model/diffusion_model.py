from dataclasses import dataclass

@dataclass
class NoiseSchedulerConfig:
    _target_: str = "emgen.generative_model.diffusion.noise_scheduler.NoiseScheduler"
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"

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
    diffusion_arch: LinearArchConfig | UNetArchConfig | UNet1DArchConfig = LinearArchConfig()
    diffusion_type: str = "ddpm"
    time_emb: str = "sinusoidal"
    input_emb: str = "sinusoidal"

    #! Training
    train_batch_size: int = 32
    eval_batch_size: int = 1000
    num_epochs: int = 200
    learning_rate: float = 1e-3
    
    #! Logging
    save_images_step: int = 1