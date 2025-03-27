import hydra
from hydra.utils import instantiate

from emgen_config.config import EmGenConfig

@hydra.main(version_base=None, config_name="config")
def main(cfg: EmGenConfig) -> None:
    print(cfg.generative_model)
    emgen = instantiate(cfg)
    print(emgen)

if __name__ == "__main__":
    main()