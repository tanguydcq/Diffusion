import torch
import os
from diffusion import Diffusion
from config import config
from model import get_model
from dataset import get_data
from utils import make_gif, save_images

import argparse
from config import get_config_by_name

# ... existing imports ...

def infer(args, config_name):
    # Determine correct paths based on config
    run_name = os.path.join(args.dataset_name, config_name)
    
    # Setup
    device = args.device
    model = get_model(args).to(device)
    
    # Load model checkpoint
    # Try finding the final model primarily
    ckpt_path = os.path.join("models", run_name, f"{model.__class__.__name__}_final.pt")
    if not os.path.exists(ckpt_path):
        # Fallback to older naming/locations if needed, for backward compat or interruptions
        print(f"Checkpoint not found at {ckpt_path}. Checking for intermediate checkpoints...")
        dir_path = os.path.join("models", run_name)
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith('.pt')]
            if files:
                # Sort to find latest if named with epoch
                files.sort() 
                ckpt_path = os.path.join(dir_path, files[-1])
                print(f"Found checkpoint: {ckpt_path}")
    
    if os.path.exists(ckpt_path):
        print(f"Loading model from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path))
    else:
        print(f"No checkpoint found in {os.path.join('models', run_name)}! Running with random weights.")
        return # Or continue if testing random weights

    model.eval()
    diffusion = Diffusion(
        img_size=args.img_size,
        img_channels=args.image_channels,
        device=device,
        noise_steps=args.T,
        beta_start=args.beta_start,
        beta_end=args.beta_end
    )

    # Output directory
    results_dir = os.path.join("results", run_name)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Noise Process GIF
    print("Generating Noise Process GIF...")
    dataloader = get_data(args)
    images, _ = next(iter(dataloader))
    images = images[:16].to(device) # Select 16 images
    
    noise_frames = []
    # Add initial clean image
    noise_frames.append(((images.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8).cpu())
    
    # Generate noise steps
    stride = 10
    for t in range(0, diffusion.noise_steps, stride):
        t_tensor = torch.tensor([t] * images.shape[0], device=device).long()
        x_t, _ = diffusion.noise_images(images, t_tensor)
        
        # Convert to visual format
        frame = ((x_t.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8).cpu()
        noise_frames.append(frame)
        
    if (diffusion.noise_steps - 1) % stride != 0:
        t = diffusion.noise_steps - 1
        t_tensor = torch.tensor([t] * images.shape[0], device=device).long()
        x_t, _ = diffusion.noise_images(images, t_tensor)
        frame = ((x_t.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8).cpu()
        noise_frames.append(frame)

    make_gif(noise_frames, os.path.join(results_dir, "noise.gif"))
    print(f"Noise GIF saved to {os.path.join(results_dir, 'noise.gif')}")

    # 2. Denoise (Sampling) Process GIF
    print("Generating Sampling (Denoise) GIF...")
    sampled_images, sample_frames = diffusion.sample(model, n=16, save_gif=True)
    
    make_gif(sample_frames, os.path.join(results_dir, "sampling.gif"))
    print(f"Sampling GIF saved to {os.path.join(results_dir, 'sampling.gif')}")

    # 3. Save sampled images
    save_images(sampled_images, os.path.join(results_dir, "sampling.jpg"), nrow=4)
    print("Sampled images saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config1', help='Configuration name')
    cli_args = parser.parse_args()
    
    try:
        config_args = get_config_by_name(cli_args.config)
        infer(config_args, cli_args.config)
    except ValueError as e:
        print(e)
