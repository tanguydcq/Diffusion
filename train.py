import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import setup_logging, save_images
from model import get_model
from config import config
import logging
from dataset import get_data
from diffusion import Diffusion
import imageio
import numpy as np

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def make_gif(frames, path):
    # Frames shape: (T, N, 3, H, W)
    import torchvision
    formatted_frames = []
    for frame_batch in frames:
        # frame_batch is (N, 3, H, W)
        # Select first 16 or less
        n_imgs = min(16, frame_batch.shape[0])
        subset = frame_batch[:n_imgs] 
        # Make grid using torchvision for better layout (default is nrow=8, we can force nrow=4 for 16 images)
        grid = torchvision.utils.make_grid(subset, nrow=int(np.sqrt(n_imgs)), padding=2, normalize=False)
        
        # Grid is (3, H_grid, W_grid). Convert to numpy (H, W, 3)
        grid = grid.permute(1, 2, 0).cpu().numpy()
        
        # Scale to uint8 if not already (input frames are typically float or uint8, diffusion.py says they are uint8)
        # diffusion.py returns uint8 frames, so no need to scale by 255.
        
        formatted_frames.append(grid)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, formatted_frames, fps=10, loop=0)

def train(args=config):
    setup_logging(run_name="DDPM")
    device = args.device
    dataloader = get_data(args)
    model = get_model(args).to(device)
    print(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(noise_steps=args.T, img_size=args.img_size, img_channels=args.image_channels, device=device)
    
    logger = logging.getLogger()
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}/{args.epochs}")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            # Select x0 (real images)
            images = images.to(device)

            # Select t (random timesteps)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            
            # Add noise to x0 to get x_t
            x_t, noise = diffusion.noise_images(images, t)
            
            # Predict noise given x_t and t
            predicted_noise = model(x_t, t)
            
            # Calculate loss
            loss = mse(noise, predicted_noise)
            
            # Backprop and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())

        sampled_images, frames = diffusion.sample(model, n=16, save_gif=True)
        save_images(sampled_images, os.path.join("results", args.dataset_name, "DDPM", f"{epoch}.jpg"))
        
        # Save GIF every 10 epochs or at the end
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            make_gif(frames, os.path.join("results", args.dataset_name, "DDPM", f"{epoch}.gif"))

        torch.save(model.state_dict(), os.path.join("models", args.dataset_name, "DDPM", f"ckpt.pt"))

if __name__ == '__main__':
    train()
