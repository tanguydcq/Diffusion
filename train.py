import os
import torch
import torch.nn as nn
import logging
import imageio
import numpy as np
import torchvision
import argparse

from tqdm import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from utils import setup_logging, save_images
from model import get_model
from dataset import get_data
from diffusion import Diffusion
from config import get_config_by_name


logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


def make_gif(frames, path):
    # Frames shape: (T, N, 3, H, W)
    formatted_frames = []
    for frame_batch in frames:
        # frame_batch is (N, 3, H, W)
        # Select first 16 or less
        n_imgs = min(16, frame_batch.shape[0])
        subset = frame_batch[:n_imgs]
        # Make grid using torchvision for better layout (default is nrow=8, we can force nrow=4 for 16 images)
        grid = torchvision.utils.make_grid(
            subset, nrow=int(np.sqrt(n_imgs)), padding=2, normalize=False
        )

        # Grid is (3, H_grid, W_grid). Convert to numpy (H, W, 3)
        grid = grid.permute(1, 2, 0).cpu().numpy()

        # Scale to uint8 if not already (input frames are typically float or uint8, diffusion.py says they are uint8)
        # diffusion.py returns uint8 frames, so no need to scale by 255.

        formatted_frames.append(grid)

    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, formatted_frames, fps=10, loop=0)


# ... existing imports ...


def train(args, config_name):
    # Create the run_name structure: e.g. MNIST/config1
    run_name = os.path.join(args.dataset_name, config_name)

    setup_logging(run_name)
    device = args.device
    dataloader = get_data(args)
    model = get_model(args).to(device)

    # TensorBoard logs in runs/MNIST/config1
    writer = SummaryWriter(os.path.join("runs", run_name))
    print(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(
        noise_steps=args.T,
        img_size=args.img_size,
        img_channels=args.image_channels,
        device=device,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )

    logger = logging.getLogger()
    l = len(dataloader)

    # Log Beta Schedule once
    writer.add_histogram(
        "Hyperparameters/Beta_Schedule", diffusion.beta.cpu(), global_step=0
    )

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

            # Calculate Gradient Norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            writer.add_scalar(
                "Training/GradNorm", total_norm, global_step=epoch * l + i
            )

            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            writer.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step=epoch)

        if epoch % 10 == 0:
            sampled_images = diffusion.sample(model, n=4)
            # Save generated images to results/MNIST/config1/
            save_path = os.path.join("results", run_name, f"sample_epoch_{epoch}.png")
            save_images(sampled_images, save_path)
            writer.add_images("Generated_Images", sampled_images, global_step=epoch)

        if epoch % 20 == 0:
            torch.save(
                model.state_dict(), os.path.join("models", run_name, f"ckpt_{epoch}.pt")
            )

    torch.save(
        model.state_dict(),
        os.path.join("models", run_name, f"{model.__class__.__name__}_final.pt"),
    )
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config1",
        help="Configuration name (e.g., config1, config2)",
    )
    cli_args = parser.parse_args()

    # Load the specific configuration
    try:
        config_args = get_config_by_name(cli_args.config)
        print(f"Loading configuration: {cli_args.config}")
        train(config_args, cli_args.config)
    except ValueError as e:
        print(e)
