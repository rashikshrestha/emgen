import hydra
from hydra.utils import instantiate
import os
import torch
import numpy as np
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
        'results',
        'hyperparameter_evaluation'
    )

    # Run KL divergence visualization
    visualize_kl_divergence(
        beta_schedules=['linear', 'cosine', 'quadratic'],
        T_values=[10, 100, 1000, 10000],
        n_samples=10000,
        output_dir=output_dir
    )

    print(f"Evaluation completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
