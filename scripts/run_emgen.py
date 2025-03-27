import hydra
from hydra.utils import instantiate

from emgen_config.config import EmGenConfig

@hydra.main(version_base=None, config_name="config")
def main(cfg: EmGenConfig) -> None:
    emgen = instantiate(cfg)
    emgen.generative_model.train()

if __name__ == "__main__":
    main()