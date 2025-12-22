import torch
import logging
import torch.nn as nn
from tqdm import tqdm
from config import config

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, img_channels=3, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.img_channels = img_channels
        self.device = device
        
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        ε = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * ε, ε
    
    def sample_timesteps(self, n):
        # Randomly sample timesteps
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def sample(self, model, n, save_gif=False):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.img_channels, self.img_size, self.img_size)).to(self.device)
            frames = []
            for i in tqdm(reversed(range(0, self.noise_steps)), position=0):
                # We generate z ~ N(0, I) if i > 1 else z = 0
                z = torch.randn_like(x) if i > 0 else torch.zeros_like(x)

                # create time batch for all images
                t = (torch.ones(n) * i).long().to(self.device)
                # predict noise 
                predicted_noise = model(x, t)

                # Reconstruction formula
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                coef1 = 1 / torch.sqrt(alpha)
                coef2 = beta / torch.sqrt(1 - alpha_hat)
                sigma = torch.sqrt(beta)

                x = coef1 * (x - coef2 * predicted_noise) + sigma * z
                
                if save_gif:
                    # Normalize to [0, 1] for visualization
                    frame = (x.clamp(-1, 1) + 1) / 2
                    frame = (frame * 255).type(torch.uint8).cpu()
                    frames.append(frame)

        model.train()
        # Normalize to [0, 1]
        x = (x.clamp(-1, 1) + 1) / 2
        
        if save_gif:
            return x, frames
        return x
