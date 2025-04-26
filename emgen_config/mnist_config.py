from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

from emgen_config.generative_model.diffusion_model import NoiseSchedulerConfig, UNetArchConfig
from emgen_config.dataset.dataset import MNISTDatasetConfig
from emgen_config.train import TrainConfig

@dataclass
class MNISTDiffusionConfig:
    _target_: str = "emgen.generative_model.diffusion.diffusion_model.DiffusionModel"
    device: str = "cuda"
    noise_scheduler: NoiseSchedulerConfig = NoiseSchedulerConfig(num_timesteps=1000)
    diffusion_arch: UNetArchConfig = UNetArchConfig(in_channels=1, base_channels=64, num_down=2)
    dataset: MNISTDatasetConfig = MNISTDatasetConfig(train=True)
    train: TrainConfig = TrainConfig(
        train_batch_size=64,
        eval_batch_size=64,
        num_epochs=50,
        learning_rate=1e-4,
        save_images_step=5
    )

@dataclass
class MNISTExperimentConfig:
    experiment_name: str = "mnist_diffusion"
    device: str = "cuda"
    generative_model: MNISTDiffusionConfig = MNISTDiffusionConfig()

# Register the configuration
cs = ConfigStore.instance()
cs.store(name="mnist_config", node=MNISTExperimentConfig)

