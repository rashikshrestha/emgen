import argparse
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from hydra.core.hydra_config import HydraConfig

#! emgen imports
from emgen.utils.visualization import plot_2d_intermediate_samples, plot_images_intermediate_samples
from emgen.generative_model.diffusion.noise_scheduler import NoiseScheduler
from emgen.generative_model.diffusion.diffusion_model_arch import LinearArch
from emgen.dataset.toy_dataset import ToyDataset
from emgen.utils.config_schema import TrainConfig

log = logging.getLogger(__name__)


class DiffusionModel():
    """
    Train and Sample from the Diffusion Model
    """
    def __init__(
        self,
        device: str = 'cpu',
        noise_scheduler: NoiseScheduler = NoiseScheduler(),
        diffusion_arch: LinearArch = LinearArch(),
        dataset: ToyDataset = ToyDataset(),
        train: TrainConfig = TrainConfig()
    ):
        #! Inputs
        self.device = device
        self.noise_scheduler = noise_scheduler
        self.diffusion_arch = diffusion_arch.to(self.device)
        self.train_config = train
        
        self.out_dir = Path(HydraConfig.get().runtime.output_dir)
        self.data_dim = dataset[0].shape

        #! DataLoader 
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.train_config.train_batch_size,
            shuffle=True,
            drop_last=True
        )
        
        #! Optimizer
        self.optimizer = torch.optim.AdamW(
            diffusion_arch.parameters(),
            lr=self.train_config.learning_rate,
        )
        
        self.frames = []
        
    
    def train(self):
        log.info("Training the Diffusion Model")
        self.diffusion_arch.train()
        
        for epoch in tqdm(range(self.train_config.num_epochs)):
            for batch in tqdm(self.train_loader):
                batch = batch.to(self.device)

                #! Sample Random Noise
                noise = torch.randn(batch.shape, device=self.device)
                
                #! Sample random timesteps
                timesteps = torch.randint(
                    0, self.noise_scheduler.num_timesteps, (batch.shape[0],),
                    device=self.device
                ).long()
                
                #! Add noise to the batch
                noisy = self.noise_scheduler.add_noise(batch, noise, timesteps)
                
                #! Use model to predict noise
                noise_pred = self.diffusion_arch(noisy, timesteps)
                
                #! Calculate Loss
                loss = F.mse_loss(noise_pred, noise)
                loss.backward()
                
                #! Update Model 
                nn.utils.clip_grad_norm_(self.diffusion_arch.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            if epoch % self.train_config.save_images_step == 0 or epoch == self.train_config.num_epochs - 1:
                sample_out = self.sample()
                sample_out_dir = self.out_dir/f"train_epoch_{epoch:02d}"
                sample_out_dir.mkdir(parents=True, exist_ok=True)
                
                #! Plotting
                # If Image Data
                if len(sample_out['intermediate_samples'].shape) == 5:
                    plot_images_intermediate_samples(
                        sample_out['intermediate_samples'], 
                        sample_out_dir, 
                        self.train_config.no_of_diff_samples_to_save,     
                    )
                # If 2D Data
                elif len(sample_out['intermediate_samples'].shape) == 3:
                    plot_2d_intermediate_samples(
                        sample_out['intermediate_samples'], 
                        sample_out_dir, 
                        self.train_config.no_of_diff_samples_to_save,     
                    )
                torch.save(self.diffusion_arch.state_dict(), self.out_dir/f"train_epoch_{epoch:02d}/diffusion_arch.pth")

 
    def sample(self, get_intermediate_samples=True):
        log.info("Sampling from the Diffusion Model")
        self.diffusion_arch.eval()
        #! Accumulators
        output = {}
        intermediate_samples = []
        #! Sample random noise
        sample = torch.randn(torch.Size([self.train_config.eval_batch_size, *self.data_dim]), device=self.device)
        # from emgen.dataset.toy_dataset import get_gt_dino_dataset
        # sample = get_gt_dino_dataset()
        # sample = torch.as_tensor(sample, device=self.device, dtype=torch.float32)
        if get_intermediate_samples: intermediate_samples.append(sample.cpu().numpy())
        #! List the timesteps in reverse order for sampling
        timesteps = list(range(len(self.noise_scheduler)))[::-1]
        #! Reverse diffusion process
        for i, t in enumerate(tqdm(timesteps)):
            t = torch.from_numpy(np.repeat(t, self.train_config.eval_batch_size)).long().to(self.device)
            # t = torch.from_numpy(np.repeat(t, 142)).long().to(self.device)
            with torch.no_grad():
                residual = self.diffusion_arch(sample, t)
            sample = self.noise_scheduler.step(residual, sample, t[0]) # Same timestep for all samples
            if get_intermediate_samples: intermediate_samples.append(sample.cpu().numpy())  # Store intermediate frames for visualization
        if get_intermediate_samples: output['intermediate_samples'] = np.array(intermediate_samples)
        #! Final sample
        generated_sample = sample.cpu().numpy()
        output['generated_sample'] = generated_sample
        return output