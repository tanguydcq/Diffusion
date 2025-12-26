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


import shutil

def train(args, config_name, resume=False, reset=False):
    # Create the run_name structure: e.g. MNIST/config1
    run_name = os.path.join(args.dataset_name, config_name)
    
    # Paths
    models_dir = os.path.join("models", run_name)
    results_dir = os.path.join("results", run_name)
    runs_dir = os.path.join("runs", run_name)

    if reset:
        print(f"Resetting training for {run_name}...")
        for path in [models_dir, results_dir, runs_dir]:
            if os.path.exists(path):
                print(f"Deleting {path}")
                shutil.rmtree(path)
        resume = False # Force resume to False if resetting
    
    # Auto-resume: if not explicitly reset and checkpoints exist, resume automatically
    if not reset and not resume:
        # Check if there are existing checkpoints
        if os.path.exists(models_dir):
            checkpoint_files = [f for f in os.listdir(models_dir) if f.startswith('ckpt_') and f.endswith('.pt')]
            if checkpoint_files:
                print(f"Existing checkpoints found in {models_dir}. Auto-resuming training...")
                resume = True
            else:
                print(f"No checkpoints found. Starting fresh training...")
        else:
            print(f"No models directory found. Starting fresh training...")

    setup_logging(run_name)
    device = args.device
    dataloader = get_data(args)
    model = get_model(args).to(device)
    
    # Note: torch.compile() requires Triton which is not easily available on Windows
    # Uncomment below if you have Triton installed:
    # try:
    #     model = torch.compile(model)
    #     print("Model compiled with torch.compile()")
    # except Exception as e:
    #     print(f"torch.compile() not available or failed: {e}")

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
    
    # Resume from checkpoint if requested
    start_epoch = 0
    global_step = 0
    if resume:
        models_dir = os.path.join("models", run_name)
        checkpoint_path = None
        latest_epoch = -1
        
        if os.path.exists(models_dir):
            # 1. Scan numbered checkpoints (ckpt_XX.pt)
            for f in os.listdir(models_dir):
                if f.startswith('ckpt_') and f.endswith('.pt') and f != 'ckpt_final.pt':
                    try:
                        epoch_num = int(f.split('_')[1].split('.')[0])
                        if epoch_num > latest_epoch:
                            latest_epoch = epoch_num
                            checkpoint_path = os.path.join(models_dir, f)
                    except ValueError:
                        continue
            
            # 2. Check final checkpoint (ckpt_final.pt)
            final_path = os.path.join(models_dir, "ckpt_final.pt")
            if os.path.exists(final_path):
                try:
                    # Peek at the epoch in final checkpoint to compare
                    # We map to cpu to avoid loading full model to gpu just for checking
                    final_ckpt = torch.load(final_path, map_location='cpu')
                    final_epoch = final_ckpt.get('epoch', -1) if isinstance(final_ckpt, dict) else -1
                    
                    # If final is newer or equal, prefer it
                    if final_epoch >= latest_epoch:
                        latest_epoch = final_epoch
                        checkpoint_path = final_path
                except Exception as e:
                    print(f"Warning: Could not check ckpt_final.pt: {e}")

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Handle both old format (just state_dict) and new format (dict with keys)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
            else:
                # Old format: just the state_dict
                model.load_state_dict(checkpoint)
                print("Loaded old format checkpoint (state_dict only), optimizer state not restored")
            
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"No valid checkpoint found in {models_dir}, starting from scratch")

    logger = logging.getLogger()
    l = len(dataloader)
    
    # Calculate global_step based on start_epoch
    global_step = start_epoch * l
    print(f"Resuming training at global_step {global_step}")

    logger = logging.getLogger()
    l = len(dataloader)

    # Log Beta Schedule once
    writer.add_histogram(
        "Hyperparameters/Beta_Schedule", diffusion.beta.cpu(), global_step=0
    )

    # CFG dropout probability (train unconditionally some percentage of the time)
    cfg_dropout = getattr(args, 'cfg_dropout', 0.1)  # Default 10% unconditional training
    print(f"CFG dropout probability: {cfg_dropout}")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        logging.info(f"Starting epoch {epoch}/{args.epochs}")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            # Select x0 (real images)
            images = images.to(device, non_blocking=True)
            
            # Handle different label formats
            # - MNIST: labels is a tensor of class indices
            # - CryptoPunks with attributes: labels is (type_idx, accessory_vector) or list [type_idx, accessory_vector]
            if isinstance(labels, (tuple, list)) and len(labels) == 2 and isinstance(labels[0], torch.Tensor) and isinstance(labels[1], torch.Tensor):
                # Multi-attribute conditioning (CryptoPunks)
                # The DataLoader collates tuples into lists of stacked tensors
                type_idx = labels[0].to(device, non_blocking=True)
                accessory_vector = labels[1].to(device, non_blocking=True)
                labels_for_model = (type_idx, accessory_vector)
            else:
                # Simple class conditioning (MNIST)
                labels = labels.to(device, non_blocking=True)
                labels_for_model = labels

            # === CFG DROPOUT ===
            # Randomly drop conditioning to train unconditionally (required for CFG at inference)
            drop_conditioning = torch.rand(1).item() < cfg_dropout

            # Select t (random timesteps)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)

            # Add noise to x0 to get x_t
            x_t, noise = diffusion.noise_images(images, t)

            # Predict noise given x_t et t (avec labels si le modèle le supporte)
            if hasattr(model, 'attr_embedding') and model.attr_embedding is not None:
                # Multi-attribute model (CryptoPunks with attributes)
                if drop_conditioning:
                    # Train unconditionally (for CFG)
                    predicted_noise = model(x_t, t, type_idx=None, accessory_vector=None)
                else:
                    type_idx, accessory_vector = labels_for_model
                    predicted_noise = model(x_t, t, type_idx=type_idx, accessory_vector=accessory_vector)
            elif hasattr(model, 'num_classes') and model.num_classes is not None:
                # Simple class conditioning (MNIST)
                if drop_conditioning:
                    # Train unconditionally (for CFG)
                    predicted_noise = model(x_t, t, y=None)
                else:
                    predicted_noise = model(x_t, t, labels_for_model)
            else:
                # Unconditional
                predicted_noise = model(x_t, t)

            # Calculate loss
            loss = mse(noise, predicted_noise)

            # Backprop and update
            optimizer.zero_grad()
            loss.backward()

            # Calculate Gradient Norm (only every 10 steps to save time)
            if i % 10 == 0:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm**0.5
                writer.add_scalar(
                    "Training/GradNorm", total_norm, global_step=global_step
                )

            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            writer.add_scalar("MSE", loss.item(), global_step=global_step)
            
            global_step += 1

        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step=epoch)

        if epoch % 10 == 0:
            sampled_images = diffusion.sample(model, n=4)
            # Save generated images to results/MNIST/config1/
            save_path = os.path.join("results", run_name, f"sample_epoch_{epoch}.png")
            save_images(sampled_images, save_path)
            writer.add_images("Generated_Images", sampled_images, global_step=epoch)

        if epoch % 2 == 0:
            save_checkpoint(model, optimizer, epoch, os.path.join("models", run_name, f"ckpt_{epoch}.pt"))

    save_checkpoint(model, optimizer, epoch, os.path.join("models", run_name, "ckpt_final.pt"))
    writer.close()


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config1",
        help="Configuration name (e.g., config1, config2)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset training: delete all checkpoints, results and logs before starting",
    )
    cli_args = parser.parse_args()

    # Load the specific configuration
    try:
        config_args = get_config_by_name(cli_args.config)
        print(f"Loading configuration: {cli_args.config}")
        train(config_args, cli_args.config, resume=cli_args.resume, reset=cli_args.reset)
    except ValueError as e:
        print(e)
