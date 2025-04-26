from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

from emgen_config.generative_model.diffusion_model import DiffusionModelConfig

@dataclass
class KLDivergenceConfig:
    beta_schedule: str = "linear"
    num_timesteps: int = 100
    n_samples: int = 10000
    output_dir: str = "results/kl_divergence"

@dataclass
class EmGenConfig:
    experiment_name: str = "base"
    device: str = "cuda"
    generative_model: DiffusionModelConfig = DiffusionModelConfig()
    kl_divergence: KLDivergenceConfig = KLDivergenceConfig()
    
cs = ConfigStore.instance()
cs.store(name="config", node=EmGenConfig)
# cs.store(group="db", name="mysql", node=MySQLConfig)
# cs.store(group="db", name="postgresql", node=PostGreSQLConfig