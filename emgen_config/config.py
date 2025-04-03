from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

from emgen_config.generative_model.diffusion_model import DiffusionModelConfig

@dataclass
class EmGenConfig:
    experiment_name: str = "base"
    device: str = "cuda"
    generative_model: DiffusionModelConfig = DiffusionModelConfig()
    
cs = ConfigStore.instance()
cs.store(name="config", node=EmGenConfig)
# cs.store(group="db", name="mysql", node=MySQLConfig)
# cs.store(group="db", name="postgresql", node=PostGreSQLConfig