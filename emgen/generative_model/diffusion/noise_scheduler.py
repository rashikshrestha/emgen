import torch
from torch.nn import functional as F
import numpy as np

class NoiseScheduler():
    def __init__(
        self,
        device: str = 'cpu',
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        deterministic_sampling=False,
        eta=0,
        pred_x0=False,
        skip=1
    ):
        self.device = device
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.deterministic_sampling = deterministic_sampling
        self.eta = eta
        self.pred_x0 = pred_x0
        self.skip = skip

        #! Betas and Alphas
        self.betas = self.get_beta_schedule(beta_schedule, beta_start, beta_end, num_timesteps)
        # print(self.betas.shape)
        self.alphas = 1.0 - self.betas
        # print(self.alphas.shape)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        # print(self.alphas_cumprod.shape)
        # print(self.alphas_cumprod[:10])
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        # print(self.alphas_cumprod_prev.shape)
        # print(self.alphas_cumprod_prev[:10])

        #! Forward Diffusion
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        #! Reverse Diffusion (reconstruct_x0)
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1 / self.alphas_cumprod - 1)

        #! Reverse Diffusion (q_posterior)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
   
        
    def __len__(self):
        return self.num_timesteps
    
    def get_beta_schedule(self, type, start, end, steps):
        if type == "linear":
            return torch.linspace(start, end, steps, dtype=torch.float32, device=self.device)
        elif type == "quadratic":
            return torch.linspace(start ** 0.5, end ** 0.5, steps, dtype=torch.float32, device=self.device) ** 2
        elif type == "cosine":
            # Implementation follows improved DDPM paper
            steps_array = torch.linspace(0, steps, steps + 1, dtype=torch.float32, device=self.device)
            alphas_cumprod = torch.cos(((steps_array / steps) + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {type}")

 
    def add_noise(self, x_start, x_noise, timesteps: torch.Tensor):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]
        s1 = s1.reshape(-1, *[1] * (x_start.dim() - 1))
        s2 = s2.reshape(-1, *[1] * (x_start.dim() - 1))
        return s1 * x_start + s2 * x_noise


    def reconstruct_x0(self, x_t: torch.Tensor, noise: torch.Tensor, t: int):
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise


    def q_posterior(self, x_0, x_t, t: int):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu


    def get_variance(self, t):
        if t == 0:
            return 0
        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance


    def step(self, x_t, eps_t_or_x_0, t: int):
        """
        Args:
            x_t (torch.Tensor): Current sample at timestep t
            eps_t_or_x_0 (torch.Tensor): Either the noise or the x_0 sample
            t (int): Current timestep
        Returns:
            x_tminus1 (torch.Tensor): Refined sample at timestep t-1
        """
        #! Check if eps_t_or_x_0 is noise or x_0
        if self.pred_x0:
            x_0 = eps_t_or_x_0
            eps_t = (x_t - self.sqrt_alphas_cumprod[t] * x_0) / self.sqrt_one_minus_alphas_cumprod[t]
        else:
            eps_t = eps_t_or_x_0
            x_0 = self.reconstruct_x0(x_t, eps_t, t)
      
        #! DDPM vs DDIM 
        if not self.deterministic_sampling: # DDPM
            x_tminus1 = self.q_posterior(x_0, x_t, t)
            variance = 0
            if t > 0:
                noise = torch.randn_like(x_t, device=self.device)
                variance = (self.get_variance(t) ** 0.5) * noise
            x_tminus1 += variance
            
        else: # DDIM
            if t > 0 and self.eta > 0:
                frac = (1 - self.alphas_cumprod_prev[t]) / (1 - self.alphas_cumprod[t])
                sigma_t = self.eta * torch.sqrt(frac * (1 - self.alphas_cumprod[t] / self.alphas_cumprod_prev[t]))
            else:
                sigma_t = torch.tensor(0.0, device=x_t.device)
               
            
            if self.skip == 1:
                dir_xt = torch.sqrt(1 - self.alphas_cumprod_prev[t] - sigma_t**2) * eps_t
                x_tminus1 = torch.sqrt(self.alphas_cumprod_prev[t]) * x_0 + dir_xt
            else:
                idx = t-self.skip
                if idx < 0:
                    idx = 0
                dir_xt = torch.sqrt(1 - self.alphas_cumprod[idx] - sigma_t**2) * eps_t
                x_tminus1 = torch.sqrt(self.alphas_cumprod[idx]) * x_0 + dir_xt
            
            if t > 0 and self.eta > 0:
                noise = torch.randn_like(x_t, device=self.device)
                x_tminus1 += sigma_t * noise


        return x_tminus1