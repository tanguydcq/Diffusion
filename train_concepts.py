import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from tqdm import tqdm
from src.model_vector import get_model
from src.diffusion import Diffusion
from src.config import get_config_by_name

def train_concept(model, diffusion, args, name, target_type=None, target_acc_idx=None, steps=1000):
    device = args.device
    # Initialisation du vecteur c (h-space: 512 canaux)
    # Pour img_size=32: 32->16->8->4 après 3 downsampling
    bottleneck_size = args.img_size // 8
    c = torch.zeros((1, 512, bottleneck_size, bottleneck_size), device=device, requires_grad=True)
    optimizer = optim.Adam([c], lr=2e-3)
    mse = nn.MSELoss()

    # 1. Génération des références x+ (Savoir interne du modèle)
    print(f"[*] Extraction du concept : {name}")
    with torch.no_grad():
        t_idx = torch.full((16,), target_type if target_type is not None else 3, dtype=torch.long, device=device)
        acc = torch.zeros(16, args.num_accessories, device=device)
        if target_acc_idx is not None: acc[:, target_acc_idx] = 1.0
        
        # Génération de 16 images de référence possédant l'attribut
        x_refs = diffusion.sample_cfg(model, n=16, type_idx=t_idx, accessory_vector=acc, guidance_scale=5.0)
        x_refs = x_refs * 2 - 1 # Normalisation [-1, 1]

    # 2. Optimisation du vecteur c (Self-Discovery)
    for i in range(steps):
        optimizer.zero_grad()
        indices = torch.randint(0, 16, (4,))
        x_0 = x_refs[indices]
        t = torch.randint(0, args.T, (4,)).to(device)
        
        x_t, noise_added = diffusion.noise_images(x_0, t)
        # Prediction via base NEUTRE (Male sans accessoires) + injection c
        noise_pred = model(x_t, t, type_idx=None, accessory_vector=None, concept_vector=c)
        
        loss = mse(noise_pred, noise_added)
        loss.backward()
        optimizer.step()

    # Sauvegarde
    torch.save(c.detach(), f"concepts/{name}.pt")

def main():
    config_name = "cryptopunks_classes"
    args = get_config_by_name(config_name)
    os.makedirs("concepts", exist_ok=True)
    
    model = get_model(args).to(args.device)
    ckpt = torch.load(f"models/CRYPTOPUNKS_CLASSES/{config_name}/ckpt_final.pt", map_location=args.device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()
    
    diffusion = Diffusion(img_size=args.img_size, device=args.device, noise_steps=args.T)

    # Extraction des Types (0-4)
    types_names = ["alien", "ape", "female", "male", "zombie"]
    for i, name in enumerate(types_names):
        train_concept(model, diffusion, args, f"type_{name}", target_type=i)

    # Extraction d'accessoires iconiques (Exemples d'index, à adapter selon votre metadata)
    iconic_acc = {"pipe": 15, "cig": 20, "shades": 30, "cap": 50, "hoodie": 60}
    for name, idx in iconic_acc.items():
        train_concept(model, diffusion, args, f"acc_{name}", target_acc_idx=idx)

if __name__ == "__main__":
    main()