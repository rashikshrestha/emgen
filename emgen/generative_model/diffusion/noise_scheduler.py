import torch
from torch.nn import functional as F


class NoiseScheduler():
    def __init__(
        self,
        device: str = 'cpu',
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        diffusion_type="ddpm"
    ):
        self.device = device
        self.num_timesteps = num_timesteps

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

 
    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_start + s2 * x_noise


    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise


    def q_posterior(self, x_0, x_t, t):
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


    def step(self, model_output, timestep, sample):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output, device=self.device)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample


torch.set_printoptions(precision=5, sci_mode=False)
noise_scheduler = NoiseScheduler()
