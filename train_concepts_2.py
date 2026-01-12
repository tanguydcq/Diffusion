import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from tqdm import tqdm

# Imports de votre framework
from src.model_vector import get_model
from src.diffusion import Diffusion
from src.config import get_config_by_name

def extract_concept(model, diffusion, args, name, target_type=None, target_acc_idx=None, steps=1000, lr=1e-3):
    """
    Optimisation stochastique d'une direction latente par reconstruction auto-supervisée.
    """
    device = args.device
    
    # Initialisation du vecteur c (h-space : 512 canaux)
    # Pour img_size=32: 32->16->8->4 après 3 downsampling
    bottleneck_size = args.img_size // 8
    c = torch.zeros((1, 512, bottleneck_size, bottleneck_size), device=device, requires_grad=True)
    optimizer = optim.Adam([c], lr=lr)
    mse = nn.MSELoss()

    # 1. Acquisition du set de référence (x+) via le savoir interne du modèle
    # On génère 16 échantillons représentatifs de l'attribut cible
    with torch.no_grad():
        t_idx = torch.full((16,), target_type if target_type is not None else 3, dtype=torch.long, device=device)
        acc = torch.zeros(16, args.num_accessories, device=device)
        if target_acc_idx is not None:
            acc[:, target_acc_idx] = 1.0
        
        # Sampling avec CFG pour garantir la pureté sémantique des références
        x_refs = diffusion.sample_cfg(model, n=16, type_idx=t_idx, accessory_vector=acc, guidance_scale=5.0)
        x_refs = x_refs * 2 - 1 # Mapping vers l'espace latent [-1, 1]

    # 2. Boucle d'optimisation (Self-Discovery)
    # On force le vecteur c à compenser l'absence de conditionnement explicite
    for i in range(steps):
        optimizer.zero_grad()
        indices = torch.randint(0, 16, (4,))
        x_0 = x_refs[indices]
        t = torch.randint(0, args.T, (4,)).to(device)
        
        x_t, noise_added = diffusion.noise_images(x_0, t)
        
        # Prédiction avec prompt neutre + injection du vecteur c dans le bottleneck
        noise_pred = model(x_t, t, type_idx=None, accessory_vector=None, concept_vector=c)
        
        loss = mse(noise_pred, noise_added)
        loss.backward()
        optimizer.step()

    # Sauvegarde de l'artefact sémantique
    save_path = f"concepts/{name}.pt"
    torch.save(c.detach(), save_path)
    return save_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='cryptopunks_classes_fast')
    parser.add_argument('--steps', type=int, default=1200)
    cli_args = parser.parse_args()

    args = get_config_by_name(cli_args.config)
    os.makedirs("concepts", exist_ok=True)
    
    # Setup modèle gelé (Frozen Master Model)
    model = get_model(args).to(args.device)
    ckpt_path = os.path.join("models", args.dataset_name, cli_args.config, "ckpt_final.pt")
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    
    diffusion = Diffusion(img_size=args.img_size, device=args.device, noise_steps=args.T)

    # --- PARTIE 1 : EXTRACTION DES TYPES (0-4) ---
    types = ["alien", "ape", "female", "male", "zombie"]
    print(f"[*] Lancement de l'extraction des {len(types)} types...")
    for i, name in enumerate(types):
        path = extract_concept(model, diffusion, args, f"type_{name}", target_type=i, steps=cli_args.steps)
        print(f"    [OK] Type {name} sauvegardé : {path}")

    # --- PARTIE 2 : EXTRACTION DES ACCESSOIRES (0-86) ---
    # Nous itérons sur la totalité des attributs définis dans le dataset
    print(f"[*] Lancement de l'extraction des {args.num_accessories} accessoires...")
    for i in range(args.num_accessories):
        # On utilise un nommage générique par index pour faciliter l'appel programmatique
        path = extract_concept(model, diffusion, args, f"acc_{i}", target_acc_idx=i, steps=cli_args.steps)
        if i % 5 == 0:
            print(f"    [PROGRESS] Accessoire {i}/{args.num_accessories} traité.")

    print("[!] Bibliothèque sémantique complète générée dans /concepts.")

if __name__ == "__main__":
    main()