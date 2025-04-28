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
from emgen.utils.visualization import plot_2d_intermediate_samples, plot_images_intermediate_samples, plot_kl_divergences, plot_timeseries_1d_pdf
from emgen.generative_model.diffusion.noise_scheduler import NoiseScheduler
from emgen.generative_model.diffusion.diffusion_model_arch import LinearArch
from emgen.dataset.toy_dataset import ToyDataset
from emgen.utils.config_schema import TrainConfig
from emgen.utils.metrics import compute_Nd_kl_divergence, project_to_lower_dim

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
        self.dataset = dataset
        self.train_config = train
        self.train_length = len(dataset)
        self.no_of_eval_batches = int(self.train_length/self.train_config.eval_batch_size)
        
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
               
            #! If this is 'save_images' epoch or the last epoch
            if epoch % self.train_config.save_images_step == 0 or epoch == self.train_config.num_epochs - 1:
                print("generating samples...")
                all_generated_samples = []
                for eb in tqdm(range(self.no_of_eval_batches)):
                    sample_out = self.sample()
                    all_generated_samples.append(sample_out['intermediate_samples'])
                    
                    if eb==0:
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
                        
                all_generated_samples = np.concatenate(all_generated_samples, axis=1)
                all_generated_samples = torch.as_tensor(all_generated_samples)
               
                all_kl = self.compute_kl(all_generated_samples, self.dataset.data) 
                
                timesteps = list(range(all_kl.shape[0]))[::-1]
                timesteps = np.array(timesteps)
                
                plot_kl_divergences(
                    all_kl, timesteps, output_file=sample_out_dir/"kl_divergences.png", 
                    title = f"Epoch {epoch:02}: KL Divergences between GT and Generated Samples",
                    x_label="Timesteps", invert_x=True
                )
               
                all_generated_samples_1d, gt_samples_1d = project_to_lower_dim(all_generated_samples, self.dataset.data, out_dim=1)
                all_generated_samples_1d, gt_samples_1d = all_generated_samples_1d.squeeze(), gt_samples_1d.squeeze()
                
                
                samples_to_plot = np.vstack((all_generated_samples_1d, gt_samples_1d))
                timesteps = np.concatenate((timesteps, np.array([0])))
                
                
                plot_timeseries_1d_pdf(samples_to_plot, timesteps,
                    title=f"Epoch {epoch:02}: Evolution of PDF during Reverse Diffusion",
                    figsize=(8, 5),
                    save_path=sample_out_dir/"pdf_1d_evolution.png"
                )

 
    def sample(self, get_intermediate_samples=True):
        # log.info("Sampling from the Diffusion Model")
        self.diffusion_arch.eval()
        #! Accumulators
        output = {}
        intermediate_samples = []
        #! Sample random noise
        sample = torch.randn(torch.Size([self.train_config.eval_batch_size, *self.data_dim]), device=self.device)
        # from emgen.dataset.toy_dataset import get_gt_dino_dataset
        # sample = get_gt_dino_dataset()
        # sample = torch.as_tensor(sample, device=self.device, dtype=torch.float32)
        if get_intermediate_samples: intermediate_samples.append(sample.cpu().numpy()) # Initial noise
        #! List the timesteps in reverse order for sampling
        timesteps = list(range(len(self.noise_scheduler)))[::-1]
        #! Reverse diffusion process
        for i, t in enumerate(timesteps):
            t = torch.from_numpy(np.repeat(t, self.train_config.eval_batch_size)).long().to(self.device)
            # t = torch.from_numpy(np.repeat(t, 142)).long().to(self.device)
            with torch.no_grad():
                residual = self.diffusion_arch(sample, t)
            sample = self.noise_scheduler.step(sample, residual, t[0]) # Same timestep for all samples
            if get_intermediate_samples: intermediate_samples.append(sample.cpu().numpy())  # Store intermediate frames for visualization
        if get_intermediate_samples: output['intermediate_samples'] = np.array(intermediate_samples)
        #! Final sample
        generated_sample = sample.cpu().numpy()
        output['generated_sample'] = generated_sample
        return output
    
    
    def compute_kl(self, all_gen_samples, gt_samples):
        """
        Compute KL divergence between the GT samples and all generated samples
        
        Args:
            gt_samples (torch.Tensor): Ground truth samples (N,d)
            all_gen_samples (torch.Tensor): All generated samples (T,N,d)
        """
        #! Convert to numpy arrays if they are PyTorch tensors
        if isinstance(gt_samples, torch.Tensor):
            gt_samples = gt_samples.detach().cpu().numpy()
        if isinstance(all_gen_samples, torch.Tensor):
            all_gen_samples = all_gen_samples.detach().cpu().numpy()
        
        #! Compute KL divergence
        all_kl = []
        for gen_samples in all_gen_samples:
            kl_div = compute_Nd_kl_divergence(gen_samples, gt_samples)
            all_kl.append(kl_div)
            
        return np.array(all_kl)