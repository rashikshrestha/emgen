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

#! emgen imports
from emgen.generative_model.diffusion.noise_scheduler import NoiseScheduler
from emgen.generative_model.diffusion.diffusion_model_arch import LinearArch
from emgen.dataset.toy_dataset import ToyDataset
from emgen_config.train import TrainConfig

log = logging.getLogger(__name__)


class DiffusionModel():
    """
    Train and Sample from the Diffusion Model
    """
    def __init__(
        self,
        noise_scheduler: NoiseScheduler = NoiseScheduler(),
        diffusion_arch: LinearArch = LinearArch(),
        dataset: ToyDataset = ToyDataset(),
        train: TrainConfig = TrainConfig()
    ):
        self.noise_scheduler = noise_scheduler
        self.diffusion_arch = diffusion_arch
        self.dataset = dataset.get_dataset()
        self.train_config = train
        
        self.optimizer = torch.optim.AdamW(
            diffusion_arch.parameters(),
            lr=self.train_config.learning_rate,
        )
        
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=self.train_config.train_batch_size,
            shuffle=True,
            drop_last=True
        )
        
        self.frames = []
        
    
    def train(self):
        log.info("Training the Diffusion Model")
        self.diffusion_arch.train()
        
        for epoch in tqdm(range(self.train_config.num_epochs)):
            for batch in self.train_loader:
                # batch is a (32,2) tensor wrapped by list. why wrapped?
                batch = batch[0]
                noise = torch.randn(batch.shape)
                timesteps = torch.randint(
                    0, self.noise_scheduler.num_timesteps, (batch.shape[0],)
                ).long()
                noisy = self.noise_scheduler.add_noise(batch, noise, timesteps)
               
                #! Use model 
                noise_pred = self.diffusion_arch(noisy, timesteps)
                
                #! Calculate Loss
                loss = F.mse_loss(noise_pred, noise)
                loss.backward()
               
                #! Update Model 
                nn.utils.clip_grad_norm_(self.diffusion_arch.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            if epoch % self.train_config.save_images_step == 0 or epoch == self.train_config.num_epochs - 1:
                print(f"LOSS: {loss}")
                self.sample()
                
        print("Saving images...")
        imgdir = f"images"
        os.makedirs(imgdir, exist_ok=True)
        frames = np.stack(self.frames)
        xmin, xmax = -6, 6
        ymin, ymax = -6, 6
        for i, frame in enumerate(frames):
            plt.figure(figsize=(10, 10))
            plt.scatter(frame[:, 0], frame[:, 1])
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.savefig(f"{imgdir}/{i:04}.png")
            plt.close()

 
    def sample(self):
        log.info("Sampling from the Diffusion Model")
        self.diffusion_arch.eval()
        sample = torch.randn(self.train_config.eval_batch_size, 2)
        timesteps = list(range(len(self.noise_scheduler)))[::-1]
        for i, t in enumerate(tqdm(timesteps)):
            t = torch.from_numpy(np.repeat(t, self.train_config.eval_batch_size)).long()
            with torch.no_grad():
                residual = self.diffusion_arch(sample, t)
            sample = self.noise_scheduler.step(residual, t[0], sample)
        self.frames.append(sample.numpy())