import hydra
from hydra.utils import instantiate
import os
from omegaconf import DictConfig, OmegaConf

from emgen.utils.visualization import visualize_kl_divergence
from emgen_config.config import EmGenConfig

@hydra.main(version_base=None, config_path="../emgen_config", config_name="config")
def main(cfg: EmGenConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Extract the device from config
    device = cfg.device

    # Create output directory
    output_dir = os.path.join(
        os.getcwd(),
        cfg.kl_divergence.output_dir
    )

    # Run KL divergence visualization
    visualize_kl_divergence(
        beta_schedule=cfg.kl_divergence.beta_schedule,
        num_timesteps=cfg.kl_divergence.num_timesteps,
        n_samples=cfg.kl_divergence.n_samples,
        output_dir=output_dir,
        device=device
    )

    print(f"Evaluation completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
