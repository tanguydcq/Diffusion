import torch
import os
from src.diffusion import Diffusion
from src.model import get_model
from src.dataset import get_data
from src.utils import make_gif, save_images

import argparse
from src.config import get_config_by_name

def infer(args, config_name):
    # Determine correct paths based on config
    run_name = os.path.join(args.dataset_name, config_name)
    
    # Setup
    device = args.device
    model = get_model(args).to(device)
    
    # Load model checkpoint
    models_dir = os.path.join("models", run_name)
    ckpt_path = None
    latest_epoch = -1

    if os.path.exists(models_dir):
        # 1. Scan numbered checkpoints (ckpt_XX.pt)
        for f in os.listdir(models_dir):
            if f.startswith('ckpt_') and f.endswith('.pt') and f != 'ckpt_final.pt':
                try:
                    epoch_num = int(f.split('_')[1].split('.')[0])
                    if epoch_num > latest_epoch:
                        latest_epoch = epoch_num
                        ckpt_path = os.path.join(models_dir, f)
                except ValueError:
                    continue
        
        # 2. Check final checkpoint (ckpt_final.pt)
        final_path = os.path.join(models_dir, "ckpt_final.pt")
        if os.path.exists(final_path):
            try:
                # Peek at the epoch in final checkpoint
                final_ckpt = torch.load(final_path, map_location='cpu')
                final_epoch = final_ckpt.get('epoch', -1) if isinstance(final_ckpt, dict) else -1
                
                # If final is newer or equal, prefer it
                if final_epoch >= latest_epoch:
                    latest_epoch = final_epoch
                    ckpt_path = final_path
            except Exception as e:
                print(f"Warning: Could not check ckpt_final.pt: {e}")
    
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading model from {ckpt_path} (Epoch {latest_epoch})")
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"No checkpoint found in {models_dir}! Running with random weights.")
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
    
    # Get CFG scale from config (default to 3.0 if not specified)
    cfg_scale = getattr(args, 'cfg_scale', 3.0)
    use_cfg = cfg_scale > 1.0
    print(f"Using CFG: {use_cfg} (scale={cfg_scale})")
    
    # Test conditionnel basé sur le type de modèle
    if hasattr(model, 'attr_embedding') and model.attr_embedding is not None:
        # Multi-attribute conditioning (CryptoPunks)
        print(f"Testing multi-attribute conditional generation...")
        print(f"  - Types: {model.num_types}")
        print(f"  - Accessories: {model.num_accessories}")
        
        # Générer 16 images: 4 par type (exemple avec types 0-3)
        # Type: 0=Alien, 1=Ape, 2=Female, 3=Male, 4=Zombie (alphabétique)
        n_samples = 16
        type_idx = torch.tensor([i % model.num_types for i in range(n_samples)]).to(device)
        
        # Créer des vecteurs d'accessoires variés (quelques exemples)
        accessory_vector = torch.zeros(n_samples, model.num_accessories).to(device)
        # Ajouter quelques accessoires aléatoires pour la variété
        for i in range(n_samples):
            # Ajouter 2-3 accessoires aléatoires par image
            num_acc = torch.randint(2, 4, (1,)).item()
            acc_indices = torch.randperm(model.num_accessories)[:num_acc]
            accessory_vector[i, acc_indices] = 1.0
        
        if use_cfg:
            sampled_images, sample_frames = diffusion.sample_cfg(
                model, n=n_samples, guidance_scale=cfg_scale, save_gif=True,
                type_idx=type_idx, accessory_vector=accessory_vector
            )
        else:
            sampled_images, sample_frames = diffusion.sample(
                model, n=n_samples, save_gif=True,
                type_idx=type_idx, accessory_vector=accessory_vector
            )
        print(f"Generated images with types: {type_idx.cpu().tolist()}")
        
    elif hasattr(model, 'num_classes') and model.num_classes is not None:
        # Simple class conditioning (MNIST)
        print(f"Testing conditional generation with {model.num_classes} classes...")
        # Générer 16 images: 4 de chaque classe (0, 1, 2, 3)
        labels = torch.tensor([i % model.num_classes for i in range(16)]).to(device)
        
        if use_cfg:
            sampled_images, sample_frames = diffusion.sample_cfg(
                model, n=16, guidance_scale=cfg_scale, save_gif=True, labels=labels
            )
        else:
            sampled_images, sample_frames = diffusion.sample(model, n=16, save_gif=True, labels=labels)
        print(f"Generated images with labels: {labels.cpu().tolist()}")
    else:
        # Génération non-conditionnelle
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
