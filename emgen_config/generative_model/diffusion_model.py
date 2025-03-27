from dataclasses import dataclass

@dataclass
class DiffusionModelConfig:
    #! Positional Embeddings
    time_emb: str = "sinusoidal",
    input_emb: str = "sinusoidal",
    
    #! Diffusion
    diffusion_type: str = "ddpm"
    timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    model_arch: str = "linear"
    
    #! Training
    train_batch_size: int = 32
    eval_batch_size: int = 1000
    num_epochs: int = 200
    learning_rate: float = 1e-3
    
    #! Logging
    save_images_step: int = 1